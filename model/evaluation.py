import pdb

import torch
from torch import nn
import numpy as np
from .utils.ssc_metric import SSCMetrics

class Evaluation(object):
    def __init__(self, args):
        self.args = args
        self.n_classes = args.num_class
        self.class_names = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                            "motorcyclist", "road",
                            "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                            "terrain", "pole", "traffic-sign", ]
        self.metrics = SSCMetrics(self.n_classes)

    def eval(self, results, result_name = 'ssc'):
        detail = dict()

        for result in results:
            self.metrics.add_batch(result['y_pred'], result['y_true'])
        metric_prefix = f'{result_name}_SemanticKITTI'
        stats = self.metrics.get_stats()
        for i, class_name in enumerate(self.class_names):
            detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

        detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
        detail["{}/IoU".format(metric_prefix)] = stats["iou"]
        detail["{}/Precision".format(metric_prefix)] = stats["precision"]
        detail["{}/Recall".format(metric_prefix)] = stats["recall"]
        self.metrics.reset()
        results = None
        return detail, stats