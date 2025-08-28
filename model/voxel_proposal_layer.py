import pdb
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from .LSS_generate import Calibration


class Voxelization(nn.Module):
    def __init__(self, point_cloud_range, spatial_shape):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = np.array([
            [point_cloud_range[0], point_cloud_range[3]],
            [point_cloud_range[1], point_cloud_range[4]],
            [point_cloud_range[2], point_cloud_range[5]]
        ])

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def filter_pc(self, pc, batch_idx):
        def mask_op(data, x_min, x_max):
            mask = (data > x_min) & (data < x_max)
            return mask
        mask_x = mask_op(pc[:, 0], self.coors_range_xyz[0][0] + 0.0001, self.coors_range_xyz[0][1] - 0.0001)
        mask_y = mask_op(pc[:, 1], self.coors_range_xyz[1][0] + 0.0001, self.coors_range_xyz[1][1] - 0.0001)
        mask_z = mask_op(pc[:, 2], self.coors_range_xyz[2][0] + 0.0001, self.coors_range_xyz[2][1] - 0.0001)
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        fiter_batch_idx = batch_idx[mask]
        if filter_pc.shape[0] < 10:
            filter_pc = torch.ones((10, 3), dtype=pc.dtype).to(pc.device)
            filter_pc = filter_pc * torch.rand_like(filter_pc)
            fiter_batch_idx = torch.zeros(10, dtype=torch.long).to(pc.device)
        return filter_pc, fiter_batch_idx

    def forward(self, pc, batch_idx):
        pc, batch_idx = self.filter_pc(pc, batch_idx)
        xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], self.spatial_shape[0])
        yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], self.spatial_shape[1])
        zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], self.spatial_shape[2])

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, _ = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)

        return unq, unq_inv


class VoxelProposalLayer(nn.Module):
    def __init__(
            self
    ):
        super(VoxelProposalLayer, self).__init__()

        self.voxelize = Voxelization(
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
            spatial_shape=np.array([128, 128, 16]))

        self.image_grid = self.create_grid()

        self.input_dimensions = [128, 128, 16]

    def create_grid(self):
        # make grid in image plane
        ogfH = 376
        ogfW = 1408
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)
        grid = torch.stack((xs, ys), 1)
        return grid

    def depth2lidar(self, image_grid, depth, cam_params, device="cuda:0"):
        depth = depth.unsqueeze(1)
        depth = depth.to(image_grid.device)
        b, _, h, w = depth.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        rots = rots.to(image_grid.device)
        trans = trans.to(image_grid.device)
        intrins = intrins.to(image_grid.device)
        post_rots = post_rots.to(image_grid.device)
        post_trans = post_trans.to(image_grid.device)
        bda = bda.to(image_grid.device)
        points = torch.cat([image_grid.repeat(b, 1, 1, 1), depth], dim=1)  # [b, 3, h, w]
        points = points.view(b, 3, h * w).permute(0, 2, 1)
        # undo pos-transformation
        points = points - post_trans.view(b, 1, 3).to(image_grid.device)
        points = torch.inverse(post_rots).view(b, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam to ego
        points = torch.cat([points[:, :, 0:2, :] * points[:, :, 2:3, :], points[:, :, 2:3, :]], dim=2)

        if intrins.shape[3] == 4:
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(b, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        combine = rots.matmul(torch.inverse(intrins)).to(torch.float32)
        points = combine.view(b, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(b, 1, 3).to(torch.float32)

        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(b, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(b, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)

        return points

    def lidar2voxel(self, points, device):
        points_reshape = []
        batch_idx = []
        tensor = torch.ones((1,), dtype=torch.long).to(device)

        for i, pc in enumerate(points):
            points_reshape.append(pc)
            batch_idx.append(tensor.new_full((pc.shape[0],), i))

        points_reshape, batch_idx = torch.cat(points_reshape), torch.cat(batch_idx)
        unq, unq_inv = self.voxelize(points_reshape, batch_idx)

        return unq, unq_inv

    def project_depth_to_points(self, calib, depth, max_high=80):
        depth = depth.squeeze(0).cpu().numpy()
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = torch.tensor(points)
        cloud = calib.project_image_to_velo(points)
        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        return cloud[valid]


    def forward(self, image_metas, depth):
        calibration = Calibration(image_metas)
        points = self.project_depth_to_points(calibration, depth).to(self.image_grid.device).unsqueeze(0)
        points = points.to(torch.float32)
        unq, unq_inv = self.lidar2voxel(points, points.device)
        sparse_tensor = spconv.SparseConvTensor(
            torch.ones(unq.shape[0], dtype=torch.float32).view(-1, 1).to(points.device),
            unq.int(), spatial_shape=self.input_dimensions, batch_size=(torch.max(unq[:, 0] + 1))
        )
        input = sparse_tensor.dense()
        return input.squeeze(0).squeeze(0)