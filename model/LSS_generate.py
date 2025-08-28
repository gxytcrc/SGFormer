import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pdb


point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [0, 40, 1],
}


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class Calibration(object):
    def __init__(self, image_metas):
        # Projection matrix from rect camera coord to image2 coord
        self.P = image_metas["cam_p"].cpu()
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        #self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = image_metas['lidar2cam'][0][:, :3, :].cpu()
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        #self.R0 = calibs['R0_rect']
        self.R0 = np.array([1,0,0,0,1,0,0,0,1])
        self.R0 = np.reshape(self.R0, [3, 3])
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return torch.from_numpy(self.project_rect_to_velo(pts_3d_rect))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

class FeaturePyramidFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidFusion, self).__init__()
        # Create convolution layers for each level to reduce dimensionality
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list])
        # Final convolution to fuse all feature maps
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)

    def forward(self, feature_maps):
        # Reduce channels for each feature map to have the same output channels
        reduced_features = [conv(fm) for conv, fm in zip(self.convs, feature_maps)]
        
        # Upsample all reduced feature maps to the same spatial size as the largest one
        target_size = reduced_features[0].shape[2:]
        upsampled_features = [F.interpolate(fm, size=target_size, mode='bilinear', align_corners=False) for fm in reduced_features]
        
        # Concatenate all feature maps along the channel dimension
        concatenated_features = torch.cat(upsampled_features, dim=1)
        
        # Apply the final fusion convolution
        fused_feature = self.fusion_conv(concatenated_features)
        
        return fused_feature

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class LSSVolume(nn.Module):
    def __init__(self, args):
        super(LSSVolume, self).__init__()
        self.dim = args.dim_num
        self.image_h = args.grd_h
        self.image_w = args.grd_w
        self.feature_fusion = FeaturePyramidFusion([self.dim,self.dim, self.dim, self.dim, self.dim], self.dim)
        self.downsample = 4
        self.grid_config = grid_config
        self.frustum = self.create_frustum()
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                        self.grid_config['ybound'],
                        self.grid_config['zbound'],
                        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)


    def generate_guassian_depth_target(self, depth, stride, cam_depth_range, constant_std=None):
        B, tH, tW = depth.shape
        kernel_size = stride
        center_idx = kernel_size * kernel_size // 2
        H = tH // stride
        W = tW // stride
        unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) #B, Cxkxk, HxW
        unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # B, H, W, kxk
        valid_mask = (unfold_depth != 0) # BN, H, W, kxk

        if constant_std is None:
            valid_mask_f = valid_mask.float() # BN, H, W, kxk
            valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
            valid_num[valid_num == 0] = 1e10
            
            mean = torch.sum(unfold_depth, dim=-1) / valid_num
            var_sum = torch.sum(((unfold_depth - mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
            std_var = torch.sqrt(var_sum / valid_num)
            std_var[valid_num == 1] = 1 # set std_var to 1 when only one point in patch
        else:
            std_var = torch.ones((B, H, W)).type_as(depth).float() * constant_std

        unfold_depth[~valid_mask] = 1e10
        min_depth = torch.min(unfold_depth, dim=-1)[0] #BN, H, W
        min_depth[min_depth == 1e10] = 0

        # x in raw depth 
        x = torch.arange(cam_depth_range[0] - cam_depth_range[2] / 2, cam_depth_range[1], cam_depth_range[2])
        # normalized by intervals
        dist = Normal(min_depth / cam_depth_range[2], std_var / cam_depth_range[2]) # BN, H, W, D
        cdfs = []
        for i in x:
            cdf = dist.cdf(i)
            cdfs.append(cdf)

        cdfs = torch.stack(cdfs, dim=-1)
        depth_dist = cdfs[..., 1:] - cdfs[...,:-1]
        return depth_dist, min_depth


    def create_frustum(self):
        # make grid in image plane
        fH, fW = self.image_h // self.downsample, self.image_w // self.downsample
        ds = torch.arange(0, 40, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, self.image_w - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, self.image_h - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum



    def voxel_pooling(self, geom_feats, x):
        geom_feats = geom_feats.to(x.device)
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        return final

    def get_geometry(self, calibration):
        points = self.frustum.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        cloud = calibration.project_image_to_velo(points).to(torch.float32)
        cloud_valid = cloud.reshape(40, self.image_h // self.downsample, self.image_w // self.downsample, 3)
        return cloud_valid.unsqueeze(0).unsqueeze(0)

    def forward(self, feature_pyramid, image_metas):
        calibration = Calibration(image_metas)
        geom = self.get_geometry(calibration)
        feature = self.feature_fusion(feature_pyramid).unsqueeze(0)
        depth = image_metas['depth'][:, :self.image_h, :self.image_w]
        depth_prob, depth = self.generate_guassian_depth_target(depth, 4, self.grid_config['dbound'], 0.5)
        depth_prob = depth_prob.permute(0, 3, 1, 2).unsqueeze(1).to(feature.device)
        feature = feature.unsqueeze(3) * depth_prob.unsqueeze(2)
        feature = feature.permute(0, 1, 3, 4, 5, 2)
        feature = self.voxel_pooling(geom, feature)
        return feature
        