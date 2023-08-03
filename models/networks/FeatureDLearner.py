
from PIL import Image
import torch
from torchvision import transforms
import random
import os
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time



def get_patches_weights(content:torch.Tensor, mask:torch.Tensor, condition:torch.Tensor, kernel, dilation, stride,
                        search_size, search_stride, detach=True):
    '''
    :param content: (N, C, H, W)
    :param condition: (N, C, H, W)
    :param patch_size:
    :param stride:
    :param search_size:
    :param detach:
    :return:
    '''
    if detach:
        content = content.detach()
        condition = condition.detach()
    patch_size = kernel + (kernel-1)*(dilation-1)
    cps = patch_size + search_stride * (search_size - 1)

    # (N,C,(H-patch_size)//stride+1,(H-patch_size)//stride+1,search_size,search_size,patch_size,patch_size)
    condition_patches = condition \
        .unfold(2, cps, stride) \
        .unfold(3, cps, stride) \
        .unfold(4, patch_size, search_stride) \
        .unfold(5, patch_size, search_stride) \
        .contiguous()

    # (N,C,(H-patch_size)//stride+1,(H-patch_size)//stride+1,search_size^2,patch_size,patch_size)
    condition_patches = condition_patches.view(*condition_patches.shape[:4], -1, *condition_patches.shape[6:])
    condition_patches_clone = condition_patches.clone()
    B, C, Lh, Lw, N, Ph, Pw = condition_patches_clone.shape

    condition_dila_patches = condition_dila_unfold(kernel, dilation, search_stride, stride, condition, cps).\
                                                   view(B, Lh, Lw, -1, C, kernel, kernel).clone()
    content_dila_patches = F.unfold(content, kernel_size=kernel, dilation=dilation, stride=stride).\
                                    permute(0, 2, 1).contiguous()
    content_dila_patches = content_dila_patches.view(B, Lh, Lw, -1, C, kernel, kernel).\
                                                expand_as(condition_dila_patches).clone()
    mask_dila_patches = F.unfold(mask, kernel_size=kernel, dilation=dilation, stride=stride).\
                                permute(0, 2, 1).contiguous()
    mask_dila_patches = mask_dila_patches.view(B, Lh, Lw, -1, C, kernel, kernel).\
                                          expand_as(condition_dila_patches).clone()

    a = torch.zeros_like(content_dila_patches)
    condition_mask_patches = torch.where(mask_dila_patches>0, condition_dila_patches, a)

    # (N,(H-patch_size)//stride+1,(H-patch_size)//stride+1,search_size^2)
    similarity = euclidean(content_dila_patches, condition_mask_patches, dim=(4, 5, 6), keepdim=True)
    # similarity = cos_theta(content_patches, condition_patches, dim=(4, 5, 6), keepdim=True)

    max_val, max_idx = torch.max(similarity, dim=3, keepdim=True)
    
    maxidx = max_idx.view(*max_idx.shape[:3])
    maxval = max_val.view(*maxidx.shape)
    maxidx = torch.where(maxval > 0, maxidx, 0)
    if maxidx.shape[2] > 32:
       num = 8
    else:
       num = 4
    w = maxidx.shape[2] // num
    swap_area = w // 2
    idx_slices = maxidx.split(w, dim=2)
    if num < len(idx_slices):
       num += 1
    mean_list = []
    for i in range(num):
       idx_s = idx_slices[i].float()
       if (i + 1) * w < maxidx.shape[2]:
           slice_area = maxidx[:, :, i * (w - swap_area):(i + 1) * w].float()
           # slice_area = maxidx[:, i * (w - swap_area):(i + 1) * w, :].float()
       else:
           slice_area = maxidx[:, :, i * (w - swap_area):maxidx.shape[2]].float()
           # slice_area = maxidx[:, i * (w - swap_area):maxidx.shape[1], :].float()
       slice_mean = torch.round(slice_area.sum() / (torch.gt(slice_area, 0.).sum() + 1e-8))
       idx_s = torch.where(idx_s > 0., idx_s, slice_mean)
       mean_list.append(idx_s)
    maxidx = torch.cat(mean_list, dim=2).type_as(max_idx)
    max_idx = maxidx.view(*max_idx.shape)
    
    max_idx = max_idx.permute(0, 4, 1, 2, 3, 5, 6)
    max_val = max_val.permute(0, 4, 1, 2, 3, 5, 6)

    return condition_patches_clone, max_idx, max_val, similarity

def swap_feature(condition_patches, max_idx, max_val, patch_size, stride, shape, count=None):
    '''

    :param condition_patches: (N,C,H',W',search_size^2,patch_size,patch_size)
    :param max_idx: (N,1,H',W',1,1,1)
    :param patch_size:
    :param stride:
    :param shape:
    :param count:
    :return:
    '''
    # (N, C, H',W', 1, patch_size, patch_size) <-- (N, 1, H',W', 1, 1, 1)
    max_idx = max_idx.repeat((1, condition_patches.shape[1], 1, 1, 1, *condition_patches.shape[5:]))
    # max_val = max_val.repeat((1, 1, 1, 1, 1, *condition_patches.shape[5:]))
    max_val = max_val.repeat((1, condition_patches.shape[1], 1, 1, 1, *condition_patches.shape[5:]))

    # (N, C, H', W', patch_size, patch_size)
    swapped_patches = condition_patches.gather(4, max_idx).squeeze(4)
    max_val = max_val.squeeze(4)
    # (N, C, patch_size, patch_size, H', W') <-- (N, C, H', W', patch_size, patch_size)
    swapped_patches = swapped_patches.permute(0, 1, 4, 5, 2, 3).contiguous()
    max_val = max_val.permute(0, 1, 4, 5, 2, 3).contiguous()
    # (N, C x patch_size x patch_size, H' x W') <-- (N, C, H', W', patch_size, patch_size)
    swapped_patches = swapped_patches.view(swapped_patches.shape[0],
                                           swapped_patches.shape[1] * swapped_patches.shape[2] * swapped_patches.shape[
                                               3], -1)
    # max_val = max_val.view(max_val.shape[0], max_val.shape[1] * max_val.shape[2] * max_val.shape[3], -1)
    max_val = max_val.view(*swapped_patches.shape)
    # (N, C, shape[0], shape[1]) <-- (N,C x patch_size x patch_size, H' x W')
    mapped_feature = F.fold(swapped_patches,
                            output_size=shape,
                            kernel_size=patch_size,
                            stride=stride)
    simi = F.fold(max_val, output_size=shape, kernel_size=patch_size, stride=stride)
    if count is None:
        count = F.fold(torch.ones_like(swapped_patches),
                       output_size=shape,
                       kernel_size=patch_size,
                       stride=stride)
        count = torch.clamp(count, min=1)
        count_s = count[:, 0:1, :, :]
    mapped_feature /= count
    simi /= count
    return mapped_feature, simi, count

def euclidean(inp1: torch.Tensor, inp2: torch.Tensor, dim, keepdim=True):
    inp1 = inp1 / (inp1.norm(p=2, dim=dim, keepdim=True) + 1e-8)
    inp2 = inp2 / (inp2.norm(p=2, dim=dim, keepdim=True) + 1e-8)
    # inp1 /= (inp1.norm(p=2, dim=dim, keepdim=True) + 1e-8)  # Warning /= is an inplace operator
    # inp2 /= (inp2.norm(p=2, dim=dim, keepdim=True) + 1e-8)
    eud = torch.norm(inp1 - inp2, p=2, dim=dim, keepdim=keepdim)

    # return 1-eud
    return (2 - eud) / 2

def condition_dila_unfold(kernel, dilation, search_stride, stride, a, cps):
    B, C, H, W = a.shape
    a_patches = F.unfold(a, kernel_size=cps, stride=stride).permute(0, 2, 1).contiguous()
    a_patches = a_patches.view(-1, C, cps, cps)
    a_patches = F.unfold(a_patches, kernel_size=kernel, dilation=dilation, stride=search_stride).permute(0, 2, 1).contiguous()
    return a_patches

def content_dila_unfold(kernel, dilation, stride, b):
    b_patches = F.unfold(b, kernel_size=kernel, dilation=dilation, stride=stride).permute(0, 2, 1).contiguous()
    b_patches = b_patches.view(-1, *b_patches.shape[-1:]).unsqueeze(1)
    return b_patches

class FeatureLeaner(nn.Module):

    def __init__(self, kernel, searchsize, stride, search_stride, dilation):
        super(FeatureLeaner, self).__init__()

        self.count = None
        self.kernel = kernel
        self.dilation = dilation
        self.patch_size = self.kernel + (self.kernel-1)*(self.dilation-1)
        self.search_size = searchsize
        self.stride = stride
        self.search_stride = search_stride
        self.count_dict = {}

    def forward(self, content, mask, condition):

        condition_patches, max_idx, weights, _ = get_patches_weights(content, mask, condition, self.kernel, self.dilation,
                                                 self.stride, self.search_size, self.search_stride)


        swapped_features, similarity, count = swap_feature(condition_patches, max_idx, weights, self.patch_size, self.stride,
                                                    content.shape[2:], self.count_dict.get(content.shape[-2:], None))
        self.count_dict[content.shape[-2:]] = count

        return swapped_features, similarity
