# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class BEVHeader(nn.Module):
    def __init__(self, class_num, norm_layer, feature):
        super(BEVHeader, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, bev_features):
        res = {}
        bev_features_up = self.up_scale(bev_features)
        bs, dim, h, w = bev_features_up.shape
        bev_features_up = bev_features_up.squeeze().permute(1, 2, 0).reshape(-1, dim)
        ssc_logit_full = self.mlp_head(bev_features_up)
        res["bev_ssc_logit"] = ssc_logit_full.reshape(h, w, self.class_num).permute(2, 0, 1).unsqueeze(0)
        res["features"] = bev_features
        return res



class HalfHeader(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(HalfHeader, self).__init__()
        self.feature = feature
        self.class_num = class_num
        # self.conv_3d = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, input_dict):

        x3d_up_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        # x3d_test = self.conv_3d(x3d_l1)

        _, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        return res


class Header(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        # self.conv_3d = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        # x3d_test = self.conv_3d(x3d_l1)

        x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        _, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        return res

class CHeader(nn.Module):
    def __init__(
            self,
            class_num,
            norm_layer,
            feature,
    ):
        super(CHeader, self).__init__()
        self.feature = feature
        self.class_num = class_num
        # self.conv_3d = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input_dict):
        res = {}

        x3d_up_l1 = input_dict["x3d"]  # [1, 64, 128, 128, 16]
        # x3d_test = self.conv_3d(x3d_l1)

        # x3d_up_l1 = self.up_scale_2(x3d_l1)  # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        _, feat_dim, w, l, h = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1, 2, 3, 0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3, 0, 1, 2).unsqueeze(0)

        return res

