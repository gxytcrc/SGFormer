import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .resnet import resnet101_multi_level, resnet50_multi_level
from .FPN import FPN, FeaturePyramidUpsample, FPNNeck
from .fusion_head import fusion_head
from .utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss, CE_prob_ssc_loss, BCE_ssc_loss
from .utils.ssc_metric import SSCMetrics
from .efficientnet import EfficientNet
import pdb

class Model(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args
        self.bs = self.args.batch_size
        self.level = args.level
        self.sat_level = args.sat_level
        self.dim = args.dim_num
        self.num_class = args.num_class

        # self.image_backbone = resnet50_multi_level(pretrained=True, level=self.level)
        # for param in self.image_backbone.parameters():
        #     param.requires_grad = False
        self.image_backbone = EfficientNet.from_pretrained('efficientnet-b7')
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        # Set all BatchNorm layers to eval mode
        # self.test_model = efficientnet_b7(pretrained=True)
        self.image_neck = FPN([48, 80, 224, 640, 2560], self.dim)
        self.sat_backbone = resnet50_multi_level(pretrained=True, level=self.sat_level)
        if self.args.pretrained_sat == 1:
            self.sat_backbone.load_state_dict(torch.load('check_point/sat_backbone_pretrain.pth', map_location=torch.device('cpu')))
            for param in self.sat_backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.sat_backbone.fc.parameters():
                param.requires_grad = False
        self.sat_neck = FPN([256, 512, 1024, 2048], self.dim)
        # self.sat_neck_test = FPN([256, 128, 64, 16], self.dim)
        # self.sat_neck = FPNNeck(self.dim)
        self.fusion_head = fusion_head(args)
        self.class_names = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                            "motorcyclist", "road",
                            "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                            "terrain", "pole", "traffic-sign", ]
        self.metrics = SSCMetrics(20)
        self.class_weights = torch.from_numpy(
            np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557,
                      0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        # self.class_weights = torch.from_numpy(
        #     np.array([0, 0.803, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557,
        #               0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        self.CE_ssc_loss = CE_ssc_loss
        self.CE_prob_ssc_loss = CE_prob_ssc_loss
        self.BCE_loss = BCE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = False
        self.alpha = 0.54
        self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level1 + 0.001)
        )
        # self.lss_generate = LSSVolume(args)        


    def Voxel_features_generate(self, input_image, sat_image, image_metas):
        # img_features = self.image_backbone(input_image)
        img_features_test = self.image_backbone.extract_endpoints(input_image)
        img_features =[img_features_test['reduction_2'],
                       img_features_test['reduction_3'],
                       img_features_test['reduction_4'],
                       img_features_test['reduction_5'],
                       img_features_test['reduction_6']]
        # print(img_features[0].shape)
        #put the features into FPN
        sat_features = self.sat_backbone(sat_image)
        # sat = sat_image[:, :, 128:384, 256:512]
        # test_features, test_conf = self.VGG_test(sat)
        # sat_pyramid = self.sat_neck_test(test_features)
        feature_pyramid = self.image_neck(img_features)
        sat_pyramid = self.sat_neck(sat_features)
        # lss_volume = self.lss_generate(feature_pyramid, image_metas)
        #print the pyramid shape
        level = 0
        for i in range(len(feature_pyramid)):
            # print("feature_pyramid, level", level, ": ", feature.shape)
            feature_pyramid[i] = feature_pyramid[i].unsqueeze(1)
            level = level + 1
        for i in range(len(sat_pyramid)):
            # print("feature_pyramid, level", level, ": ", feature.shape)
            sat_pyramid[i] = sat_pyramid[i].unsqueeze(1)
            level = level + 1
        output = self.fusion_head(feature_pyramid, sat_pyramid, image_metas)
        return output


    def evaluate(self, pred, gt):
        ssc_pred = pred["ssc_logit"]
        y_true = gt.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)

    def loss_cal(self, pred, gt):
        ssc_pred =  pred["ssc_logit"]
        loss_dict = dict()
        class_weight = self.class_weights.type_as(gt)
        loss_ssc = self.CE_ssc_loss(ssc_pred, gt, class_weight)
        loss_dict['loss_ssc'] = loss_ssc
        loss_sem_scal, empty_loss, class_loss = self.sem_scal_loss(ssc_pred, gt)
        loss_dict['loss_sem_scal'] = loss_sem_scal
        loss_dict['class_loss'] = class_loss
        loss_dict['empty_loss'] = empty_loss
        loss_geo_scal = self.geo_scal_loss(ssc_pred, gt)
        loss_dict['loss_geo_scal'] = loss_geo_scal
        return loss_dict

    def loss_cal_full(self, pred, gt, gt_2d, prob_2d, gt_down):
        "calculate the full size loss"
        ssc_pred =  pred["ssc_logit"]
        loss_dict = dict()
        class_weight = self.class_weights.type_as(gt)
        loss_ssc = self.CE_ssc_loss(ssc_pred, gt, class_weight)
        loss_dict['loss_ssc'] = loss_ssc
        loss_sem_scal, empty_loss, class_loss = self.sem_scal_loss(ssc_pred, gt)
        loss_dict['loss_sem_scal'] = loss_sem_scal
        loss_dict['class_loss'] = class_loss
        loss_dict['empty_loss'] = empty_loss
        loss_geo_scal = self.geo_scal_loss(ssc_pred, gt)
        loss_dict['loss_geo_scal'] = loss_geo_scal
        "calculate the bev loss"
        bev_pred = pred["bev_ssc_logit"]
        # sat_pred = pred["sat_ssc_logit"]
        loss_ssc_bev = self.CE_prob_ssc_loss(bev_pred, prob_2d, gt_2d, class_weight)
        # loss_ssc_sat = self.CE_prob_ssc_loss(sat_pred, prob_2d, gt_2d, class_weight)
        loss_dict['loss_bev'] =  loss_ssc_bev #+ 0.1 *loss_ssc_sat
        "calculate the up sampling loss"
        "add coarse loss"
        gt[gt == 255] = -1
        coarse_gt = torch.max_pool3d(kernel_size=(2, 2, 2), input=gt)
        gt[gt == -1] = 255
        coarse_gt[coarse_gt == -1] = 255
        coarse_pred = pred["coarse_ssc_logit"]
        # coarse_pred_0 = pred["coarse_ssc_logit_0"]
        coarse_loss_ssc = self.CE_ssc_loss(coarse_pred, coarse_gt, class_weight)
        # coarse_loss_ssc_0 = self.CE_ssc_loss(coarse_pred_0, coarse_gt, class_weight)
        loss_dict['loss_ssc'] = loss_dict['loss_ssc'] + 0.25*coarse_loss_ssc# + 0.1 * coarse_loss_ssc_0
        return loss_dict

    def output_pred(self, pred, gt):
        ssc_pred = pred["ssc_logit"]
        ssc_pred = ssc_pred.unsqueeze(-1)
        y_true = gt.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        result = dict()
        result['y_pred'] = y_pred
        result['y_true'] = y_true
        return result

    def forward(self, grd_image, sat_image, image_meta, gt, gt_2d, prob_2d, status, gt_down = None):
        pred = self.Voxel_features_generate(grd_image, sat_image, image_meta)
        # loss_dict = self.loss_cal(pred, gt)
        if status == "train":
            loss_dict = self.loss_cal_full(pred, gt, gt_2d, prob_2d, gt_down)
            return loss_dict
        if status == "eval":
            result = self.output_pred(pred, gt)
            return result




