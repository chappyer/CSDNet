"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random, torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, load_size, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)
    
    if opt.isTrain:
        x = random.randint(0, np.maximum(0, new_w - load_size))
        y = random.randint(0, np.maximum(0, new_h - load_size))
        center_x = x + load_size // 2
        if new_h < load_size:
            center_y = new_h // 2
        else:
            center_y = y + load_size // 2
    else:
        center_x = new_w // 2
        center_y = new_h // 2

    flip = random.random() > 0.5
    return {'crop_pos':(center_x , center_y), 'flip':flip}


def get_transform(opt, params, imgsize, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []

    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], imgsize)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    cent_x, cent_y = pos
    x1 = cent_x - size[0] // 2
    x2 = cent_x + size[0] // 2
    y1 = cent_y - size[1] // 2
    y2 = cent_y + size[1] // 2
    return img.crop((x1, y1, x2, y2))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def CoarseMatch(opt, content_tensor, sem_tensor, condition_tensor):
    C, H, W = content_tensor.shape
    patch_size = opt.coarse_patch_size
    maxsize_w = opt.coarse_maxsize_w
    maxsize_h = opt.coarse_maxsize_h
    search_stride = opt.coarse_stride
    coarse_match = torch.randn(2, H, W).fill_(float('nan'))
    dis_cx = []
    wn = opt.coarse_num_w
    hn = opt.coarse_num_h
    ws = (W-patch_size//2) // wn
    hs = (H-patch_size//2) // hn

    while (hn - 1) * hs + patch_size*3/2 > H:
        hn = hn - 1
    while (wn - 1) * ws + patch_size*3/2 > W:
        wn = wn - 1
    for i in range(hn):
        for j in range(wn):
            center = [j*ws+patch_size, i*hs+patch_size]
            cont = content_tensor[:, (center[1]-patch_size//2):(center[1]+patch_size//2),
                   (center[0]-patch_size//2):(center[0]+patch_size//2)]
            sem = sem_tensor[:, (center[1]-patch_size//2):(center[1]+patch_size//2),
                   (center[0]-patch_size//2):(center[0]+patch_size//2)]
            if center[0] < maxsize_w:
                cond_wl = 0
            else:
                cond_wl = center[0] - maxsize_w
            if (W-center[0]) < maxsize_w:
                cond_wr = W
            else:
                cond_wr = center[0] + maxsize_w
            if center[1] < maxsize_h:
                cond_ht = 0
            else:
                cond_ht = center[1] - maxsize_h
            if (H-center[1]) < maxsize_h:
                cond_hb = H
            else:
                cond_hb = center[1] + maxsize_h
            cond = condition_tensor[:, cond_ht:cond_hb, cond_wl:cond_wr]
            cond_patches = cond.unfold(1, patch_size, search_stride).unfold(2, patch_size, search_stride).contiguous()
            ch, cw = cond_patches.shape[1:3]
            cond_patches = cond_patches.view(*cond_patches.shape[0:1], -1, *cond_patches.shape[-2:])
            cond_patches = cond_patches.permute(1, 0, 2, 3)
            cont = cont.unsqueeze(0)
            sem = sem.unsqueeze(0)
            try:
                cond_patches = torch.where(sem>0., cond_patches, torch.tensor(0.))
            except:
                print('****')
            similarity = cos_theta(cont, cond_patches, dim=(1, 2, 3), keepdim=True)

            max_val, max_idx = torch.max(similarity, dim=0, keepdim=True)

            idx_num = int(max_idx.view(-1))
            idx_h = idx_num // cw
            idx_w = idx_num % cw
            simi_area_cx = cond_wl + idx_w * search_stride + patch_size // 2
            simi_area_cy = cond_ht + idx_h * search_stride + patch_size // 2
            if max_val == 0.:
                coarse_match[:, center[1]:(center[1]+1), center[0]:(center[0]+1)] = \
                    torch.tensor([float('nan'), float('nan')]).unsqueeze(1).unsqueeze(2)

                dis_cx.append(float('nan'))
            else:
                coarse_match[:, center[1]:(center[1]+1), center[0]:(center[0]+1)] = \
                    torch.tensor([simi_area_cx-center[0], simi_area_cy-center[1]]).unsqueeze(1).unsqueeze(2)
                dis_cx.append(simi_area_cx - center[0])

    dis_cx = torch.tensor(dis_cx, dtype=torch.float32)
    threshold, discx= ThreeSigmod(dis_cx, sigma=1.5)
    error = list(filter(lambda s:(s<threshold[0])|(s>threshold[1]), discx))
    for err in error:
        coarse_match = torch.where(coarse_match[0:1, :, :] == err, torch.tensor(float('nan')), coarse_match)

    return coarse_match

def cos_theta(inp1: torch.Tensor, inp2: torch.Tensor, dim, keepdim=True):
    inp1 = inp1 / (inp1.norm(p=2, dim=dim, keepdim=True) + 1e-8)
    inp2 = inp2 / (inp2.norm(p=2, dim=dim, keepdim=True) + 1e-8)

    return (inp1 * inp2).sum(dim, keepdim=keepdim)

def ThreeSigmod(value, sigma=3):
    a = []
    for i in value:
        if not i.isnan():
            a.append(i)
    dis = torch.tensor(a, dtype=torch.float32)
    avg = torch.mean(dis)
    std = torch.std(dis)
    threshold_up = avg + sigma*std
    threshold_down = avg - sigma*std
    return [threshold_down, threshold_up], dis

def get_coarse_params(params, dist, crop_size):
    cent_x, cent_y = params['crop_pos']
    x1 = cent_x - crop_size[0] // 2
    x2 = cent_x + crop_size[0] // 2
    y1 = cent_y - crop_size[1] // 2
    y2 = cent_y + crop_size[1] // 2
    area = dist[:, y1:y2, x1:x2]
    move_dist = torch.div(area.nansum(dim=1).sum(dim=1), torch.where(area.isnan(), 0, 1).sum(dim=1).sum(dim=1)+1e-8,
                          rounding_mode='trunc')
    cent_x = cent_x + int(move_dist[0])
    cent_y = cent_y + int(move_dist[1])

    return {'crop_pos': (cent_x, cent_y), 'flip': params['flip']}