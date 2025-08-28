import torch
from torch import nn
import numpy as np
from .utils.header import Header, BEVHeader, CHeader
from .transformer.fusion_encoder import FusionEncoder
from .transformer.FFN import Decoder, Unet
from .transformer.swin_transformer import SwinTransformer
from .transformer.generalizedfpn import GeneralizedLSSFPN
from .transformer.resnet_3d import CustomResNet3D
from .utils.modules import PositionalEncoding
from .transformer.spatial_cross_attention import DeformableAttention
from .transformer.spatial_self_attention import SpatialSelfAttention
import os
from .unet.unet_model import UNet2D, UNet3D
from .fusion import CBAMV2
from model.transformer.weight_init import xavier_init, constant_init
from .voxel_proposal_layer import VoxelProposalLayer
import pdb

class fusion_head(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(fusion_head, self).__init__()
        self.args = args
        self.bev_w = args.BEV_W
        self.bev_h = args.BEV_H
        self.real_w = args.real_W
        self.real_h = args.real_H
        self.grd_w = args.grd_w
        self.grd_h = args.grd_h
        self.bev_z = args.height_num
        self.dim = args.dim_num
        self.num_class = args.num_class
        "init the pos query of the full size"
        # self.pos_embedding = nn.Embedding(self.bev_w * self.bev_h * self.bev_z, self.dim)
        self.vox_cross_pos = PositionalEncoding(self.dim, max_len=self.bev_w * self.bev_h * self.bev_z)
        self.vox_self_pos = PositionalEncoding(self.dim, max_len=self.bev_w * self.bev_h * self.bev_z)
        self.vox_refine_pos = PositionalEncoding(self.dim, max_len=self.bev_w * self.bev_h * self.bev_z)
        self.vox_embedding = nn.Embedding(self.bev_w * self.bev_h * self.bev_z, self.dim)
        size = [self.bev_w, self.bev_h, self.bev_z]
        self.cross_encoder = FusionEncoder(args, 'cross', size, 3)
        self.refine_encoder = FusionEncoder(args, 'refine', size, 1)
        self.self_encoder = FusionEncoder(args, 'self', size, 1)
        self.full_header = Header(self.num_class, nn.BatchNorm3d, feature=self.dim)
        self.coarse_header = CHeader(self.num_class, nn.BatchNorm3d, feature=self.dim)
        self.full_mask = nn.Embedding(1, self.dim)

        "init the level, camera embedding"
        self.grd_feature_levels = args.level
        self.sat_feature_levels = args.sat_level
        self.grd_level_embeds = nn.Parameter(torch.Tensor(self.grd_feature_levels, self.dim))
        self.sat_level_embeds = nn.Parameter(torch.Tensor(self.sat_feature_levels, self.dim))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.args.number_of_cam, self.dim))

        "bev queries initilize"
        size_bev = [self.bev_w, self.bev_h, self.bev_w]
        self.bev_header = BEVHeader(self.num_class, nn.BatchNorm2d, feature=self.dim)
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.dim)
        self.bev_cross_pos = PositionalEncoding(self.dim, max_len=self.bev_w * self.bev_h)
        self.bev_self_pos = PositionalEncoding(self.dim, max_len=self.bev_w * self.bev_h)
        self.bev_encoder = FusionEncoder(args, 'bev', size_bev, 6)
        self.bev_decoder = Decoder(self.dim * 2, self.dim * 2, self.dim)
        self.depth2voxel = VoxelProposalLayer()        
        self.unet_2d = UNet2D(128, 128)
        # self.fov_unet_2d = UNet2D(128, 128)
        self.unet_3d = UNet3D(128, 128)
        self.fuse = CBAMV2(self.dim*2, self.dim)
        # self.unet_fusion = UNetFusion(128, 128)
        self.init_weight()

    def init_weight(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, DeformableAttention) or isinstance(m, SpatialSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        torch.nn.init.normal_(self.grd_level_embeds)
        torch.nn.init.normal_(self.sat_level_embeds)
        torch.nn.init.normal_(self.cams_embeds)
        # constant_init(self.full_coeff, val=1, bias=0)
        # constant_init(self.full_coeff_bev, val=1, bias=0)

    def feature_pyramid_flatten(self, feature_pyramid, status):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(feature_pyramid):
            bs, cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) #cam, bs, l, c
            if status == "grd":
                feat = feat + self.cams_embeds[:, None, None, :]
                feat = feat + self.grd_level_embeds[None, None,  lvl:lvl + 1, :].to(feat.dtype)
            elif status == "sat":
                feat = feat + self.sat_level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device= feature_pyramid[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return feat_flatten, spatial_shapes, level_start_index

    def forward(self, grd_pyramid, sat_pyramid, image_metas, lss_volume=None):
        batch_size, cam, channels, _, _ = grd_pyramid[0].shape
        dtype = grd_pyramid[0].dtype
        "flatten the feature_pyramid"
        sat_feat_flatten, sat_spatial_shapes, sat_level_start_index = self.feature_pyramid_flatten(sat_pyramid, "sat")
        grd_feat_flatten, grd_spatial_shapes, grd_level_start_index = self.feature_pyramid_flatten(grd_pyramid, "grd")
        sat_feat_flatten = sat_feat_flatten.permute(0, 2, 1, 3)
        grd_feat_flatten = grd_feat_flatten.permute(0, 2, 1, 3)
        feat_flatten = [grd_feat_flatten,sat_feat_flatten]
        spatial_shapes = [grd_spatial_shapes, sat_spatial_shapes]
        level_start_index = [grd_level_start_index, sat_level_start_index]
        "=========================prepare pre-occupy=================================================================="
        # proposal = image_metas['proposal'].reshape(self.bev_w, self.bev_h, self.bev_z).permute(2, 1, 0).cpu()
        depth = image_metas['depth']
        proposal = self.depth2voxel(image_metas, depth).permute(2, 1, 0).cpu()
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        "=========================prepare bev query, and bev positional embedding====================================================="
        bev_query = self.bev_embedding.weight.to(dtype).unsqueeze(1).repeat(1, batch_size, 1)
        bev_pos_cross = self.bev_cross_pos(torch.zeros_like(bev_query))
        bev_pos_self = self.bev_self_pos(torch.zeros_like(bev_query))
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos_cross = bev_pos_cross.permute(1, 0, 2)
        bev_pos_self = bev_pos_self.permute(1, 0, 2)
        "encode the bev"
        pos_bev = [bev_pos_cross, bev_pos_self, bev_pos_cross, bev_pos_self]
        "=========================vox level attention======================================"
        vox_query = self.vox_embedding.weight.to(dtype).unsqueeze(1).repeat(1, batch_size, 1)
        vox_pos_cross = self.vox_cross_pos(torch.zeros_like(vox_query))
        vox_pos_self = self.vox_self_pos(torch.zeros_like(vox_query))
        vox_query = vox_query.permute(1, 0, 2)
        vox_pos_cross = vox_pos_cross.permute(1, 0, 2)
        vox_pos_self = vox_pos_self.permute(1, 0, 2)
        # if lss_volume is not None:
        #     lss_volume = lss_volume.flatten(2).permute(0, 2, 1)
        #     vox_query = vox_query + lss_volume
        vox_query = self.cross_encoder(image_metas, vox_query, vox_pos_cross, grd_feat_flatten,
                                       grd_feat_flatten, grd_spatial_shapes, grd_level_start_index,
                                       dtype=torch.float, unmask = unmasked_idx)
        vox_query[:, masked_idx, :] = self.full_mask.weight.view(1, self.dim).expand(masked_idx.shape[1],self.dim).to(dtype)
        # vox_query[:, masked_idx, :] = self.mlp_prior(lss_volume[:, masked_idx, :])
        vox_query = self.self_encoder(image_metas, vox_query, vox_pos_self, grd_feat_flatten,
                grd_feat_flatten, grd_spatial_shapes, grd_level_start_index,
                dtype=torch.float)
        vox_query = vox_query.reshape(batch_size, self.bev_z, self.bev_h, self.bev_w, self.dim).permute(0, 3, 2, 1, 4) #BS, W, H, Z, D
        vox_features = vox_query.permute(0, 4, 1, 2, 3) #[bs, dim, w, h, z]
        "============================================BEV attention with vox guidence=========================================================="
        vox_bev_features = torch.max_pool3d(kernel_size=(1, 1, 16), input=vox_features)
        vox_bev_features_flatten = vox_bev_features.flatten(2).permute(0, 2, 1)
        vox_bev_features = vox_bev_features.permute(0, 2, 3, 4, 1)
        bev_query_sat = self.bev_encoder(image_metas, bev_query, pos_bev, feat_flatten,
                                                feat_flatten, spatial_shapes, level_start_index,
                                                dtype=torch.float, voxbev=vox_bev_features_flatten)
        bev_features = bev_query_sat.reshape(batch_size, self.bev_h, self.bev_w, self.dim).permute(0, 3, 2, 1).unsqueeze(-1) #[bs, dim, w, h, 1]
        bev_features = bev_features.permute(0, 2, 3, 4, 1)
        bev_fused = torch.cat((vox_bev_features, bev_features), dim=-1)
        bev_fused = self.bev_decoder(bev_fused)
        bev_fused =  bev_fused.permute(0, 4, 1, 2, 3).squeeze(-1)
        "==========================BEV Conv==============================================="
        bev_fused_features = self.unet_2d(bev_fused)
        bev_output = self.bev_header(bev_fused_features)
        "============================Vox Conv==============================================="
        vox_refined_features = self.unet_3d(vox_features)
        "============================Fuse Features==========================================================="
        out_features = self.fuse(vox_refined_features, bev_fused_features)
        coarse_dict = {"x3d": out_features, }
        output_coarse = self.coarse_header(coarse_dict)
        output_coarse_prob = output_coarse["ssc_logit"]
        "=====================================refinment============================================================================"
        probabilities = torch.nn.functional.softmax(output_coarse_prob, dim=1)
        # Calculate the entropy along the second dimension
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)
        x_coords = torch.arange(entropy.shape[1])
        y_coords = torch.arange(entropy.shape[2])
        z_coords = torch.arange(entropy.shape[3])
        # Create a meshgrid of x, y, z coordinates
        x_grid, y_grid, z_grid = torch.meshgrid([x_coords, y_coords, z_coords])
        # Flatten the coordinate grids
        x_flat = x_grid.permute(2, 1, 0).flatten()
        y_flat = y_grid.permute(2, 1, 0).flatten()
        z_flat = z_grid.permute(2, 1, 0).flatten()
        xyz_coordinates = torch.stack([x_flat, y_flat, z_flat], dim=1)
        entropy = entropy.permute(0, 3, 2, 1).flatten(1)
        topk_values, topk_indices = torch.topk(entropy, 10000)
        topk_idx = np.asarray(topk_indices.cpu()).astype(np.int32)
        query = out_features.permute(0, 1, 4, 3, 2).flatten(2).permute(0, 2, 1)
        pos = self.vox_refine_pos(torch.zeros_like(query.permute(1, 0 ,2)))
        pos = pos.permute(1, 0, 2)
        query = self.refine_encoder(image_metas, query, pos, grd_feat_flatten,
                                    grd_feat_flatten, grd_spatial_shapes, grd_level_start_index,
                                    dtype=torch.float, unmask=topk_idx)
        query = query.reshape(batch_size, self.bev_z, self.bev_h, self.bev_w, self.dim).permute(0, 3, 2, 1, 4) #BS, W, H, Z, D
        fine_features = query.permute(0, 4, 1, 2, 3) + out_features#[bs, dim, w, h, z]
        "==============================================Semantic Header=========================================================="
        input_dict = {"x3d": fine_features, }
        output = self.full_header(input_dict)
        output["coarse_ssc_logit"] = output_coarse["ssc_logit"]
        output["bev_ssc_logit"] = bev_output["bev_ssc_logit"]
        return output





