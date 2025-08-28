import torch
from torch import nn
import numpy as np
import copy
import warnings
from model.transformer.spatial_cross_attention import SpatialCrossAttention
from model.transformer.spatial_self_attention import SpatialSelfAttention
from model.transformer.spatial_self_attention_3d import SpatialSelfAttention3D
from model.transformer.FFN import FFN
from model.transformer.transformer import TransformerLayer
import pdb


class FusionEncoder(nn.Module):
    def __init__(self, args, status, size, number_layers):  # device='cuda:0',
        super(FusionEncoder, self).__init__()
        self.args = args
        self.pc_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
        self.sample_num = args.height_num
        self.dim = args.dim_num
        self.status = status
        self.sat_lenth = args.sat_W
        self.bev_w = size[0]
        self.bev_h = size[1]
        self.bev_z = size[2]
        self.self_size = int(pow(size[0] * size[1] * size[2], 0.5))
        if status == "refine":
            self.num_layer = number_layers
            self.layers_grd = nn.ModuleList([VoxCrossLayer(args) for _ in range(self.num_layer)])
        if status == "cross":
            self.num_layer = number_layers
            self.layers_grd = nn.ModuleList([VoxCrossLayer(args) for _ in range(self.num_layer)])
        elif status == "self":
            self.num_layer = number_layers
            self.layers_self = nn.ModuleList([VoxSelfLayer(args, self.self_size) for _ in range(self.num_layer)])
        if status == "bev":
            self.num_layer = number_layers
            self.layers_sat = nn.ModuleList([VoxBEVLayer(args, status="sat") for _ in range(self.num_layer)])
        if status == "bev_self":
            self.num_layer = number_layers
            self.layers_self = nn.ModuleList([BEVSelfLayer(args) for _ in range(self.num_layer)])
        if status == "instence":
            self.num_layer = number_layers
            self.layers_sat = nn.ModuleList([InstenceBEVLayer(args, status="sat") for _ in range(self.num_layer)])


    def ref_points(self, H, W, Z, bs = 1, device = "cuda:0", dtype = torch.float, dim='3d'):

        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(Z, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(Z, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(Z, H, W) / H
            "stack to the shape [Z, H, W, 3]"
            ref_3d = torch.stack((xs, ys, zs), -1)
            "reshape to [Z, H, W, 3]"
            ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0).unsqueeze(0)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            "reshape to [bs, pillar, Z*H*W, 3]"
            return ref_3d

        if dim == '2d':
            H = W = self.self_size
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                          torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
            ref_y = ref_y.reshape(-1)[None]/H
            ref_x = ref_x.reshape(-1)[None]/W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        
        if dim == '3dCustom':
            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, Z - 0.5, Z, dtype=dtype, device=device),
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_z = ref_z.reshape(-1)[None] / Z
            ref_2d = torch.stack((ref_x, ref_y, ref_z), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


        if dim == 'bev':
            num_points_in_pillar = 1
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            "stack to the shape [Z, H, W, 3]"
            ref_3d = torch.stack((xs, ys, zs), -1)
            "reshape to [Z, H*W, 3]"
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        if dim  == 'bev_self':
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                          torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
            ref_y = ref_y.reshape(-1)[None]/H
            ref_x = ref_x.reshape(-1)[None]/W
            "with shape [H, W, 2], the axis coordinates follows x, y"
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


    def grd_sampling(self, reference_points, img_metas, device):
        pc_range = self.pc_range
        lidar2img = (img_metas['lidar2img'])
        filename = (img_metas['img_filename'])
        lidar2img = lidar2img[0].unsqueeze(0).to(reference_points.device) # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        grd_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas['img_shape'][0][1].to(device) #x
        reference_points_cam[..., 1] /= img_metas['img_shape'][0][0].to(device) #y

        grd_mask = (grd_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        grd_mask = grd_mask.new_tensor(np.nan_to_num(grd_mask.cpu().numpy()))
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        grd_mask = grd_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, grd_mask

    def sat_sampling(self, reference_points, device):
        pc_range = self.pc_range
        num_cam = 1
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0] # W or x-axis
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1] # H or y-axis
        "unnecessary actually, z axis is not used in airier map"
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2] # Z or z-axis
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        reference_points = reference_points.view(
            D, B, 1, num_query, 3).repeat(1, 1, num_cam, 1, 1)
        reference_points_cam = reference_points[..., 0:2] / 0.2 + int(self.sat_lenth/2)
        reference_points_cam[..., 0] /= self.sat_lenth
        reference_points_cam[..., 1] /= self.sat_lenth
        if self.args.pretrained_sat == 0:
            reference_points_cam[..., 1] = 1 - reference_points_cam[..., 1]
        sat_mask = ((reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        sat_mask = sat_mask.new_tensor(np.nan_to_num(sat_mask.cpu().numpy()))
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        sat_mask = sat_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, sat_mask

    def forward(self, image_metas, query, pos, key, value, spatial_shapes, level_start_index, dtype=torch.float, unmask = None, fb_mask = None,
                instence_query=None, instence_pos=None, voxbev=None):
        if self.status == "cross":
            self.ref_3d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='3d', device=query.device)
            if unmask is not None:
                unmask_3d = self.ref_3d[:, :, torch.from_numpy(unmask[0]).cuda().long(), :].clone()
                unmask_pos = pos[:, torch.from_numpy(unmask[0]).cuda().long(), :]
                unmask_query = query[:, torch.from_numpy(unmask[0]).cuda().long(), :]
            grd_ref, grd_mask = self.grd_sampling(unmask_3d, image_metas, unmask_3d.device)
            for lid, layer in enumerate(self.layers_grd):
                if unmask is not None and fb_mask is not None:
                    unmask_query, query_map = layer(unmask_query, unmask_pos, key, value, grd_ref, grd_mask, spatial_shapes, level_start_index, fb_mask=fb_mask)
                elif unmask is not None:
                    unmask_query = layer(unmask_query, unmask_pos, key, value, grd_ref, grd_mask,spatial_shapes, level_start_index)
                else:
                    query = layer(query, pos, key, value, grd_ref, grd_mask, spatial_shapes, level_start_index)
            query = torch.empty((query.shape[0], self.bev_z * self.bev_h * self.bev_w, self.dim), device=query.device)
            query[:, torch.from_numpy(unmask[0]).long(), :] = unmask_query[:, :, :]
            if fb_mask is not None:
                query_map_full = torch.ones((query.shape[0], self.bev_z * self.bev_h * self.bev_w), device=query.device)
                query_map_full[:, torch.from_numpy(unmask[0]).long()] = query_map[:, 0, :]
                return query, query_map_full
            return query

        if self.status == "refine":
            self.ref_3d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='3d', device=query.device)
            if unmask is not None:
                unmask_3d = self.ref_3d[:, :, torch.from_numpy(unmask[0]).cuda().long(), :].clone()
                unmask_pos = pos[:, torch.from_numpy(unmask[0]).cuda().long(), :]
                unmask_query = query[:, torch.from_numpy(unmask[0]).cuda().long(), :]
            grd_ref, grd_mask = self.grd_sampling(unmask_3d, image_metas, unmask_3d.device)
            for lid, layer in enumerate(self.layers_grd):
                if unmask is not None and fb_mask is not None:
                    unmask_query, query_map = layer(unmask_query, unmask_pos, key, value, grd_ref, grd_mask, spatial_shapes, level_start_index, fb_mask=fb_mask)
                elif unmask is not None:
                    unmask_query = layer(unmask_query, unmask_pos, key, value, grd_ref, grd_mask,spatial_shapes, level_start_index)
                else:
                    query = layer(query, pos, key, value, grd_ref, grd_mask, spatial_shapes, level_start_index)
            query[:, torch.from_numpy(unmask[0]).long(), :] = unmask_query[:, :, :]
            if fb_mask is not None:
                query_map_full = torch.ones((query.shape[0], self.bev_z * self.bev_h * self.bev_w), device=query.device)
                query_map_full[:, torch.from_numpy(unmask[0]).long()] = query_map[:, 0, :]
                return query, query_map_full
            return query


        elif self.status == "self":
            # self.ref_2d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='3dCustom', device=query.device)
            # bs, len, num_level, _ = self.ref_2d.shape
            # hybird_ref_2d = torch.stack([self.ref_2d, self.ref_2d], 1).reshape(bs * 2, len, num_level, 3)
            # for lid, layer in enumerate(self.layers_self):
            #     query = layer(query, pos, hybird_ref_2d)
            # return query

            self.ref_2d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='2d', device=query.device)
            bs, len, num_level, _ = self.ref_2d.shape
            hybird_ref_2d = torch.stack([self.ref_2d, self.ref_2d], 1).reshape(bs * 2, len, num_level, 2)
            for lid, layer in enumerate(self.layers_self):
                query = layer(query, pos, hybird_ref_2d)
            return query

        elif self.status == "bev":
            self.ref_3d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='bev', device=query.device)
            self.ref_2d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim="bev_self", device=query.device)
            sat_ref, sat_mask = self.sat_sampling(self.ref_3d, self.ref_3d.device)
            for lid, layer in enumerate(self.layers_sat):
                query = layer(query, key[1], value[1], pos[2], pos[3] ,
                                  sat_mask, sat_ref, self.ref_2d, spatial_shapes[1], level_start_index[1], voxbev=voxbev)
            return query

        elif self.status == "bev_self":
            self.ref_2d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim="bev_self", device=query.device)
            for lid, layer in enumerate(self.layers_self):
                pdb.set_trace()
                query = layer(query, pos, self.ref_2d)
            return query

        elif self.status == "instence":
            self.ref_3d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim='bev', device=query.device)
            self.ref_2d = self.ref_points(self.bev_h, self.bev_w, self.bev_z, dim="bev_self", device=query.device)
            sat_ref, sat_mask = self.sat_sampling(self.ref_3d, self.ref_3d.device)
            for lid, layer in enumerate(self.layers_sat):
                query = layer(query, key, value, pos, pos ,
                                  sat_mask, sat_ref, self.ref_2d, spatial_shapes, level_start_index, instence_query, instence_pos)
            return query
            


class VoxCrossLayer(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(VoxCrossLayer, self).__init__()
        self.args = args
        self.dim = args.dim_num
        self.cross_attention = SpatialCrossAttention(args)
        self.FFN = FFN(self.dim, 1024)
        self.NormLayer0 = nn.LayerNorm(self.dim)
        self.NormLayer1 = nn.LayerNorm(self.dim)

    def forward(self, query, pos, key, value, ref_points, vox_mask, spatial_shapes, level_start_index, attn_masks = None, fb_mask = None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.args.cross_layer_num)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.args.cross_layer_num)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.self.args.cross_layer_num, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.self.args.cross_layer_num}'
        if fb_mask is not None:
            query, query_map = self.cross_attention(
                query,
                key,
                value,
                ref_points=ref_points,
                vox_mask=vox_mask,
                query_pos=pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                fb_mask=fb_mask)
        else:
            query = self.cross_attention(
                query,
                key,
                value,
                ref_points=ref_points,
                vox_mask=vox_mask,
                query_pos=pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                fb_mask=fb_mask)
        query = self.NormLayer0(query)
        query = self.FFN(query)
        query = self.NormLayer1(query)
        if fb_mask is not None:
            return query, query_map
        return query

class VoxSelfLayer(nn.Module):
    def __init__(self, args, self_size):  # device='cuda:0',
        super(VoxSelfLayer, self).__init__()
        self.args = args
        self.dim = args.dim_num
        self.self_attention = SpatialSelfAttention(args)
        self.FFN = FFN(self.dim, 512)
        self.NormLayer0 = nn.LayerNorm(self.dim)
        self.NormLayer1 = nn.LayerNorm(self.dim)
        self.self_size = self_size

    def forward(self, query, vox_pos, ref_2d, attn_masks = None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.args.cross_layer_num)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.args.cross_layer_num)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.args.self_layer_num, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.args.self_layer_num}'
        query = self.self_attention(
            query,
            query,
            query,
            query_pose=vox_pos,
            # spatial_shapes=torch.tensor([[16, 128, 128]], device=query.device),
            spatial_shapes=torch.tensor([[self.self_size, self.self_size]], device=query.device),
            reference_points=ref_2d,
            level_start_index=torch.tensor([0], device=query.device)
        )
        query = self.NormLayer0(query)
        query = self.FFN(query)
        query = self.NormLayer1(query)

        return query

class VoxBEVLayer(nn.Module):
    def __init__(self, args, status="sat"):  # device='cuda:0',
        super(VoxBEVLayer, self).__init__()
        self.args = args
        self.dim = args.dim_num
        if status == "sat":
            self.level = args.sat_level
        else:
            self.level = args.level
        self.self_attention = SpatialSelfAttention(args)
        self.cross_attention = SpatialCrossAttention(args, level = self.level)
        self.FFN = FFN(self.dim, 1024)
        self.NormLayer0 = nn.LayerNorm(self.dim)
        self.NormLayer1 = nn.LayerNorm(self.dim)
        self.NormLayer2 = nn.LayerNorm(self.dim)
        self.bev_w = args.BEV_W
        self.bev_h = args.BEV_H
        self.bev_z = args.height_num

    def forward(self, query, key, value, query_pos, vox_pos, vox_mask, ref_points, ref_2d,
                spatial_shapes, level_start_index, attn_masks = None, voxbev=None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.args.self_layer_num)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.args.self_layer_num)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.args.self_layer_num, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.args.self_layer_num}'
        query = self.self_attention(
            query,
            query,
            query,
            query_pose=vox_pos,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            reference_points=ref_2d,
            level_start_index=torch.tensor([0], device=query.device),
            voxbev=voxbev
        )
        query = self.NormLayer0(query)
        query = self.cross_attention(
            query,
            key,
            value,
            ref_points=ref_points,
            vox_mask=vox_mask,
            query_pos=query_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.NormLayer1(query)
        query = self.FFN(query)
        query = self.NormLayer2(query)
        return query


class InstenceBEVLayer(nn.Module):
    def __init__(self, args, status="sat"):  # device='cuda:0',
        super(InstenceBEVLayer, self).__init__()
        self.args = args
        self.dim = args.dim_num
        if status == "sat":
            self.level = args.sat_level
        else:
            self.level = args.level
        self.intence_attention = SpatialCrossAttention(args)
        self.intence_bev_attention = TransformerLayer(self.dim, 8)
        self.self_attention = SpatialSelfAttention(args)
        self.cross_attention = SpatialCrossAttention(args, level = self.level)
        self.FFN = FFN(self.dim, 512)
        self.FFN1 = FFN(self.dim, 256)
        self.FFN2 = FFN(self.dim, 256)
        self.NormLayer0 = nn.LayerNorm(self.dim)
        self.NormLayer1 = nn.LayerNorm(self.dim)
        self.NormLayer2 = nn.LayerNorm(self.dim)
        self.NormLayer3 = nn.LayerNorm(self.dim)
        self.NormLayer4 = nn.LayerNorm(self.dim)
        self.NormLayer5 = nn.LayerNorm(self.dim)
        self.NormLayer6 = nn.LayerNorm(self.dim)
        self.bev_w = args.BEV_W
        self.bev_h = args.BEV_H
        self.bev_z = args.height_num

    def forward(self, query, key, value, query_pos, vox_pos, vox_mask, ref_points, ref_2d,
                spatial_shapes, level_start_index, instence_query, instence_pos, attn_masks = None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.args.self_layer_num)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.args.self_layer_num)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.args.self_layer_num, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.args.self_layer_num}'
        instance_mask = torch.ones(1, 50).to(query.device)
        instance_ref = torch.zeros(1, 50, 2).to(query.device) + 0.5
        instance_mask = instance_mask.unsqueeze(1).unsqueeze(-1)
        instance_mask = instance_mask>0
        instance_ref = instance_ref.unsqueeze(1).unsqueeze(3)
        instence_query = self.intence_attention(
            instence_query,
            key[0],
            value[0],
            ref_points=instance_ref,
            vox_mask=instance_mask,
            query_pos=instence_pos,
            spatial_shapes=spatial_shapes[0],
            level_start_index=level_start_index[0])
        instence_query = self.NormLayer3(instence_query)
        instence_query = self.FFN1(instence_query)
        instence_query = self.NormLayer6(instence_query)
        query = self.intence_bev_attention(query, instence_query, instence_query, vox_pos[3])
        query = self.NormLayer4(query)
        query = self.FFN2(query)
        query = self.NormLayer5(query)
        query = self.self_attention(
            query,
            query,
            query,
            query_pose=vox_pos[3],
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            reference_points=ref_2d,
            level_start_index=torch.tensor([0], device=query.device)
        )
        query = self.NormLayer0(query)
        query = self.cross_attention(
            query,
            key[1],
            value[1],
            ref_points=ref_points,
            vox_mask=vox_mask,
            query_pos=query_pos[2],
            spatial_shapes=spatial_shapes[1],
            level_start_index=level_start_index[1])
        query = self.NormLayer1(query)
        query = self.FFN(query)
        query = self.NormLayer2(query)
        return query


class BEVSelfLayer(nn.Module):
    def __init__(self, args, status="bev_self"):  # device='cuda:0',
        super(BEVSelfLayer, self).__init__()
        self.args = args
        self.dim = args.dim_num
        self.self_attention = SpatialSelfAttention(args)
        self.FFN = FFN(self.dim, 1024)
        self.NormLayer0 = nn.LayerNorm(self.dim)
        self.NormLayer1 = nn.LayerNorm(self.dim)
        self.bev_w = args.BEV_W
        self.bev_h = args.BEV_H

    def forward(self, query, vox_pos, ref_2d, attn_masks = None):
        if attn_masks is None:
            attn_masks = [None for _ in range(self.args.self_layer_num)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.args.self_layer_num)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                        f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.args.self_layer_num, f'The length of ' \
                                                    f'attn_masks {len(attn_masks)} must be equal ' \
                                                    f'to the number of attention in ' \
                f'operation_order {self.args.self_layer_num}'
        query = self.self_attention(
            query,
            query,
            query,
            query_pose=vox_pos,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            reference_points=ref_2d,
            level_start_index=torch.tensor([0], device=query.device)
        )
        query = self.NormLayer0(query)
        query = self.FFN(query)
        query = self.NormLayer1(query)
        return query