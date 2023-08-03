
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
from options.train_options import TrainOptions



def get_patches_weights(content: torch.Tensor, mask: torch.Tensor, condition_list, content_semantic: torch.Tensor,
                        condition_semantic_list, use_semantic, patch_size, stride,
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
        condition_list = condition_list.detach()
        if use_semantic:
            content_semantic = content_semantic.detach()
            condition_semantic_list = condition_semantic_list.detach()

    condition_list_num = condition_list.shape[1] // 3


    cps = patch_size + search_stride * (search_size - 1)

    condition_patches = condition_list[:,:3,:,:] \
        .unfold(2, cps, stride) \
        .unfold(3, cps, stride) \
        .unfold(4, patch_size, search_stride) \
        .unfold(5, patch_size, search_stride) \
        .contiguous()

    condition_patches = condition_patches.view(*condition_patches.shape[:4], -1, *condition_patches.shape[6:])

    if condition_list_num > 1:

        for idx in range(condition_list_num-1):
            sta_idx = 3 + 3 * idx
            end_idx = 6 + 3 * idx
            condition_patches_single = condition_list[:,sta_idx:end_idx,:,:]\
                .unfold(2, cps, stride) \
                .unfold(3, cps, stride) \
                .unfold(4, patch_size, search_stride) \
                .unfold(5, patch_size, search_stride) \
                .contiguous()

            condition_patches_single = condition_patches_single.view(*condition_patches_single.shape[:4], -1,
                                                                     *condition_patches_single.shape[6:])

            condition_patches = torch.cat([condition_patches, condition_patches_single], dim=4)

    condition_patches_clone = condition_patches.clone()

    if use_semantic:

        condition_semantic_patches = condition_semantic_list[:,:1,:,:] \
            .unfold(2, cps, stride) \
            .unfold(3, cps, stride) \
            .unfold(4, patch_size, search_stride) \
            .unfold(5, patch_size, search_stride) \
            .contiguous()  #

        condition_semantic_patches = condition_semantic_patches.view(*condition_semantic_patches.shape[:4], -1,
                                                                     *condition_semantic_patches.shape[6:])

        if condition_list_num > 1:

            for idx in range(condition_list_num-1):
                bef_idx = 1 + 1*idx
                aft_idx = 2 + 1*idx
                condition_semantic_patches_single = condition_semantic_list[:,bef_idx:aft_idx,:,:] \
                    .unfold(2, cps, stride) \
                    .unfold(3, cps, stride) \
                    .unfold(4, patch_size, search_stride) \
                    .unfold(5, patch_size, search_stride) \
                    .contiguous()  #

                condition_semantic_patches_single = condition_semantic_patches_single.view(
                    *condition_semantic_patches_single.shape[:4], -1, *condition_semantic_patches_single.shape[6:])

                condition_semantic_patches = torch.cat([condition_semantic_patches, condition_semantic_patches_single],
                                                       dim=4)

        content_semantic_patches = content_semantic \
            .unfold(2, patch_size, stride) \
            .unfold(3, patch_size, stride) \
            .unsqueeze(4).expand_as(condition_semantic_patches).clone()

        condition_semantic_patches = condition_semantic_patches.permute(0, 2, 3, 4, 1, 5, 6)
        content_semantic_patches = content_semantic_patches.permute(0, 2, 3, 4, 1, 5, 6)

    content_patches = content \
        .unfold(2, patch_size, stride) \
        .unfold(3, patch_size, stride) \
        .unsqueeze(4).expand_as(condition_patches).clone()

    mask_patches = mask.unfold(2, patch_size, stride) \
        .unfold(3, patch_size, stride) \
        .unsqueeze(4).expand_as(content_patches).clone()

    condition_patches = condition_patches.permute(0, 2, 3, 4, 1, 5, 6)

    content_patches = content_patches.permute(0, 2, 3, 4, 1, 5, 6)

    mask_patches = mask_patches.permute(0, 2, 3, 4, 1, 5, 6)

    mask_patches_one_channel = mask_patches[:, :, :, :, 0, :, :].unsqueeze(4)

    zero_like_content_three_channel = torch.zeros_like(content_patches)
    zero_like_content_one_channel = zero_like_content_three_channel[:, :, :, :, 0, :, :].unsqueeze(4)

    condition_mask_patches = torch.where(mask_patches > 0, condition_patches, zero_like_content_three_channel)

    similarity_1 = euclidean(content_patches, condition_mask_patches, dim=(4, 5, 6),
                             keepdim=True)

    if use_semantic:
        condition_semantic_mask_patches = torch.where(mask_patches_one_channel > 0, condition_semantic_patches,
                                                      zero_like_content_one_channel)

        similarity_2 = euclidean(content_semantic_patches, condition_semantic_mask_patches, dim=(4, 5, 6),
                                 keepdim=True)

        similarity = 0.3 * similarity_1 + 0.7 * similarity_2

    else:
        similarity = similarity_1

    max_val, max_idx = torch.max(similarity, dim=3, keepdim=True)

    maxidx = max_idx.view(*max_idx.shape[:3])
    maxval = max_val.view(*maxidx.shape)
    maxidx = torch.where(maxval > 0, maxidx, 0)
    if maxidx.shape[2] > 32:
        num = 16
    else:
        num = 8
    w = maxidx.shape[2] // num
    swap_area = w // 2
    idx_slices = maxidx.split(w, dim=2)
    # if num < len(idx_slices):
    #    num += 1
    len_slice = len(idx_slices)
    mean_list = []
    for i in range(len_slice):
        idx_s = idx_slices[i].float()
        if (i + 1) * w < maxidx.shape[2]:
            slice_area = maxidx[:, :, i * (w - swap_area):(i + 1) * w].float()
        else:
            slice_area = maxidx[:, :, i * (w - swap_area):maxidx.shape[2]].float()
        slice_mean = torch.round(slice_area.sum() / (torch.gt(slice_area, 0.).sum() + 1e-8))
        idx_s = torch.where(idx_s > 0., idx_s, slice_mean)
        mean_list.append(idx_s)
    maxidx = torch.cat(mean_list, dim=2).type_as(max_idx)
    max_idx = maxidx.view(*max_idx.shape)

    max_idx = max_idx.permute(0, 4, 1, 2, 3, 5, 6)  # max_idx: tensor(10,1,14,14,1,1,1)
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
    max_val = max_val.view(*swapped_patches.shape)

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


class FeatureLeaner(nn.Module):

    def __init__(self, patchsize, searchsize, stride, search_stride,use_semantic):
        super(FeatureLeaner, self).__init__()


        self.use_semantic = use_semantic
        self.count = None
        self.patch_size = patchsize
        self.search_size = searchsize
        self.stride = stride
        self.search_stride = search_stride
        self.count_dict = {}

    def forward(self, content, mask, condition_list ,content_semantic,condition_semantic_list):



        condition_patches, max_idx, weights, _ = get_patches_weights(content, mask, condition_list,content_semantic,condition_semantic_list,self.use_semantic, self.patch_size, self.stride,
                        self.search_size, self.search_stride, detach=True)

        swapped_features, similarity, count = swap_feature(condition_patches, max_idx, weights, self.patch_size,
                                                           self.stride,
                                                           content.shape[2:],
                                                           self.count_dict.get(content.shape[-2:], None))
        self.count_dict[content.shape[-2:]] = count

        return swapped_features, similarity
