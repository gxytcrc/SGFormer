import pdb

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import glob
import pdb

from torch.utils.data import DataLoader
from torchvision import transforms

root_dir = '/mnt/nas_8/datasets/voxformer_data' # '../../data/Kitti' # '../Data' #'..\\Data' #
ox_root_dir = '/mnt/nas_8/datasets/voxformer_data/dataset/sequences'
test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satellite'
grdimage_dir = 'ground'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 2


raw_data_sequence = ["2011_10_03_drive_0027_sync", "2011_10_03_drive_0042_sync", "2011_10_03_drive_0034_sync", "2011_09_30_drive_0016_sync",
                     "2011_09_30_drive_0018_sync", "2011_09_30_drive_0020_sync", "2011_09_30_drive_0027_sync", "2011_09_30_drive_0033_sync",
                     "2011_09_30_drive_0034_sync"]
odo_data_sequence = ["00", "01", "02", "04", "05", "06", "07", "09", "10"]
eval_data_sequence = ["08"]


class train_data(Dataset):
    def __init__(self, root, ox_root, odo_sequences,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10, args = None):

        """"generate the shift for ground-arier image"""
        shift_range_lat = 0
        shift_range_lon = 0
        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters
        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.satmap_dir = satmap_dir

        """"generate the voxel scan"""
        self.sequences = odo_sequences
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m

        self.img_W = 1220
        self.img_H = 370

        self.depthmodel = 'msnet3d'
        self.vox_root = root
        self.ox_root = ox_root
        self.nsweep = str(10)
        self.query_tag =  'query_iou5203_pre7712_rec6153'

        self.poses = self.load_poses()
        self.load_scans()
        self.target_frames = []

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        color_jitter = (0.4, 0.4, 0.4)
        self.color_jitter = (transforms.ColorJitter(*color_jitter) if color_jitter else None)
        self.eval_range = args.eval_range

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.vox_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.vox_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.vox_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            # glob_path = os.path.join(
            #     self.vox_root, "dataset", "sequences_" + self.depthmodel + "_sweep"+ self.nsweep, sequence, "queries", "*." + self.query_tag
            # )
            glob_path = os.path.join(
                self.vox_root, "dataset", "my_geo", sequence, "*." + "query"
            )
            
            for proposal_path in sorted(glob.glob(glob_path)):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "proposal_path": proposal_path
                    }
                )

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def read_SemKITTI(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = self.unpack(bin)
        return bin

    def read_occupancy_SemKITTI(self, path):
        occupancy = self.read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
        return occupancy

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def get_meta_info(self, scan, sequence, frame_id, proposal_path):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            proposal_path (str): proposal path.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.vox_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        depth_path = os.path.join(self.vox_root, "dataset", "sequences_" + self.depthmodel + "_sweep" + self.nsweep, sequence, "voxels", frame_id+".pseudo")
        # print(rgb_path)
        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []
        # print(scan["proposal_path"])
        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P = scan["P"]
        cam_k = P[0:3, 0:3]
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ lidar2cam_rt)

        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        "======================================="
        cam2lidar = torch.Tensor(lidar2cam_rt).inverse()
        rot = cam2lidar[:3, :3]
        tran = cam2lidar[:3, 3]
        crop = torch.Tensor([0, 0, self.img_W,self.img_H])
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        post_rot *= torch.Tensor([0., 0.])
        post_tran-=torch.Tensor(crop[:2])
        A = self.get_rot(0 / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot2 = A.matmul(post_rot)
        post_tran2 = A.matmul(post_tran) + b
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        bda = torch.eye(4).float()
        mlp_input = [rot, tran, intrinsic, post_rot, post_tran, bda]
        "======================================="
        depth_path = "/mnt/nas_8/datasets/voxformer_data/dataset/depth/" + sequence + "/depth/" + frame_id + ".npy"
        depth = np.load(depth_path)
        # for reference img
        seq_len = len(self.poses[sequence])
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.vox_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(np.linalg.inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)
            
            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt)
            cam_intrinsics.append(intrinsic)
            image_paths.append(rgb_path)

        proposal_bin = self.read_occupancy_SemKITTI(proposal_path)
        pseudo_pc_bin = self.read_occupancy_SemKITTI(depth_path)
        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            proposal=proposal_bin,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            cam_p = P,
            img_shape = [(self.img_H,self.img_W)],
            pseudo_pc = pseudo_pc_bin,
            depth = depth
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.vox_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.vox_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            img_ref = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img_ref = self.color_jitter(img_ref)
            # PIL to numpy
            img_ref = np.array(img_ref, dtype=np.float32, copy=False) / 255.0
            img_ref = img_ref[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img_ref))

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
        ori_image = self.normalize_rgb(img).clone()

        return ori_image, image_tensor, rgb_path

    def generate_sat_image(self, sequence, frame_id):
        SatMap_name = os.path.join(self.vox_root, "dataset", self.satmap_dir, sequence, frame_id + ".png")
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
        oxts_file_name = os.path.join(self.ox_root, sequence,"oxts", "data", frame_id + ".txt")
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        gt_shift_x = 0
        gt_shift_y = 0
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        # theta = np.random.uniform(-1, 1)
        theta = 0
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
        return sat_map

    def get_gt_info(self, sequence, frame_id):
        target_1_path = os.path.join(self.vox_root, "dataset", "labels", sequence, frame_id + "_1_1.npy")
        # print(target_1_path)
        target = np.load(target_1_path)
        # short-range groundtruth
        if self.eval_range == 25.6:
            target[128:, :, :] = 255
            target[:, :64, :] = 255
            target[:, 192:, :] = 255

        elif self.eval_range == 12.8:
            target[64:, :, :] = 255
            target[:, :96, :] = 255
            target[:, 160:, :] = 255

        target_2_path = os.path.join(self.vox_root, "dataset", "labels", sequence, frame_id + "_1_2.npy")
        target_2 = np.load(target_2_path)
        target_2 = target_2.astype(np.float32)

        return target, target_2

    def __getitem__(self, idx):
        # print("++++++++++++++++++++")
        scan = self.scans[idx]
        proposal_path = scan["proposal_path"]
        sequence = scan["sequence"]
        filename = os.path.basename(proposal_path)
        frame_id = os.path.splitext(filename)[0]
        meta_dict = self.get_meta_info(scan, sequence, frame_id, proposal_path)
        ori_img, img_list, rgb_path = self.get_input_info(sequence, frame_id)
        target, target_2 = self.get_gt_info(sequence, frame_id)
        raw_dara_id = "0000" + frame_id
        sat_img = self.generate_sat_image(sequence, raw_dara_id)
        final_tensor = torch.zeros(1)
        # print(target_path)
        # print(rgb_path)
        # print(proposal_path)
        # print("++++++++++++++++++++")
        return ori_img, sat_img, meta_dict, img_list, target, final_tensor




def load_train_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, args = None, world_size=None, rank=None):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    train_set = train_data(root=root_dir, ox_root=ox_root_dir, odo_sequences=odo_data_sequence,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range,
                              args = args)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, num_replicas=world_size, rank=rank)
    if world_size==None:
        sampler = DistributedSampler(dataset=train_set, num_replicas=1, rank=0)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=batch_size,sampler=sampler, pin_memory=False,
                              num_workers=num_thread_workers, drop_last=True)
    return train_loader

def load_eval_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, args = None, world_size=None, rank=None):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    eval_set = train_data(root=root_dir, ox_root=ox_root_dir, odo_sequences=eval_data_sequence,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range,
                              args = args)
    if world_size == None:
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, pin_memory=False,
                                  num_workers=num_thread_workers, drop_last=False)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=eval_set, num_replicas=world_size, rank=rank, shuffle=False)
        eval_loader = DataLoader(eval_set, batch_size=batch_size, sampler=sampler, pin_memory=False,
                                  num_workers=num_thread_workers, drop_last=False)
    return eval_loader