"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.FeatureLearner import FeatureLeaner
from options.train_options import TrainOptions

# from models.networks.FeatureDLearner import FeatureLeaner

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):   #out_channels: 128s
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer): # layer: Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(2,2))
        nonlocal norm_type            # norm_type: spectralinstance
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):] # subnorm_type: instance

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, opt):
        super().__init__()
        self.opt = opt
        label_nc = opt.semantic_nc
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))


        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        self.register_parameter('a', nn.Parameter(torch.ones(1)))
        self.register_parameter('b', nn.Parameter(torch.ones(1)))

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        simi_nc = 1

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_simi = nn.Sequential(
            nn.Conv2d(simi_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        # self.mlp_theta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, input, condition_list, content_semantic, condition_semantic_list, params):

        kernel_size = int(params[0].item())
        search_size = int(params[1].item())
        stride = int(params[2].item())
        search_stride = int(params[3].item())
        # dilation = int(params[4].item())
        search_pad = search_stride * (search_size - 1) // 2

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x) #normalized: [6, 1024, 8, 8]

        input = F.interpolate(input, size=x.size()[2:], mode='nearest')
        content, mask = input.chunk(2, dim=1)
        ma = torch.ones_like(mask)
        mb = torch.zeros_like(mask)
        mask = torch.where(mask>0, ma, mb)
        h, w = x.size()[2:]

        condition_list = F.interpolate(condition_list, size=(h + 2 * search_pad, w + 2 * search_pad))
        if self.opt.use_semantic:
            content_semantic = F.interpolate(content_semantic,size=x.size()[2:], mode='nearest')

            condition_semantic_list = F.interpolate(condition_semantic_list, size=(h + 2 * search_pad, w + 2 * search_pad))

        feature_learner = FeatureLeaner(kernel_size, search_size, stride, search_stride,self.opt.use_semantic)

        feature_corr, similarity = feature_learner(content, mask, condition_list, content_semantic, condition_semantic_list)

        # for weight_through_cnn and cBN
        similarity = similarity[:, 0:1, :, :]
        actv_simi = self.mlp_simi(similarity)
        actv_content = self.mlp_shared(content)
        actv_swapped = self.mlp_shared(feature_corr)
        gamma = self.mlp_gamma(actv_simi)
        alpha = self.mlp_beta(actv_content)
        beta = self.mlp_beta(actv_swapped)
        # theta = self.mlp_theta(actv_swapped)
        # out = normalized * (1+theta)+ alpha * gamma + beta
        out = normalized + alpha * gamma + beta

        return out

