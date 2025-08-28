# This is a sample Python script.
import argparse
from kitti_dataset import *
from model.models import Model
from utils import *
from model.evaluation import Evaluation
from model.utils.ssc_metric import SSCMetrics
import tracemalloc

n_classes = 20
class_names = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                    "motorcyclist", "road",
                    "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                    "terrain", "pole", "traffic-sign", ]
metrics = SSCMetrics(20)

def eval(net_eval, args, device="cuda:0"):
    mini_batch = args.batch_size
    eval_loader = load_eval_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range, args = args)
    net_eval.load_state_dict(torch.load(f"check_point/semanticKITTI.pth"))
    count = 0
    detail = dict()
    for Loop, Data in enumerate(eval_loader, 0):
        gd_image, sat_map, image_meta, image_list, gt = [item for item in Data[:-1]]
        gd_image = gd_image.to(device)
        sat_map = sat_map.to(device)
        gt = gt.to(device)
        "============================================================================================"
        bev_gt, bev_prob = bev_compress(gt, args.num_class)
        bev_gt = bev_gt.unsqueeze(0).type_as(gt)
        bev_prob = bev_prob.unsqueeze(0).type_as(gt)
        result = net_eval(gd_image, sat_map, image_meta, gt, bev_gt, bev_prob, "eval")
        print(
            f"\revaluate value: {count} / {len(eval_loader)}",
            end="")
        metrics.add_batch(result['y_pred'], result['y_true'])
        count = count + 1
    metric_prefix = 'ssc_SemanticKITTI'
    stats = metrics.get_stats()
    
    for i, class_name in enumerate(class_names):
        detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

    detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
    detail["{}/IoU".format(metric_prefix)] = stats["iou"]
    detail["{}/Precision".format(metric_prefix)] = stats["precision"]
    detail["{}/Recall".format(metric_prefix)] = stats["recall"]
    metrics.reset()
    print(f"\n{detail}")
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')  # 1e-2
    parser.add_argument('--num_cam', type=int, default=1, help='total number of camera used')

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dropout', type=int, default=0, help='0 or 1')
    parser.add_argument('--pretrained_sat', type=int, default=1, help='0 or 1')
    parser.add_argument('--level', type=int, default=5, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--dim_num', type=int, default=128, help='feature dimension')

    parser.add_argument('--BEV_W', type=int, default=128, help='width of BEV features')
    parser.add_argument('--BEV_H', type=int, default=128, help='Hight of BEV features')
    parser.add_argument('--real_W', type=float, default=51.2, help='Width of BEV in meters')
    parser.add_argument('--real_H', type=float, default=51.2, help='Hight of BEV in meters')
    parser.add_argument('--sat_W', type=int, default=512, help='Width of BEV in meters')
    parser.add_argument('--sat_H', type=int, default=512, help='Hight of BEV in meters')


    parser.add_argument('--number_of_cam', type=int, default=1, help='the camera number in the dataset')
    parser.add_argument('--map_scale', type=int, default=0.2, help='meter per pixel')
    parser.add_argument('--height_num', type=int, default=16, help='sample points number in height')
    parser.add_argument('--GPU_use', type=bool, default=True, help='use GPU?')
    parser.add_argument('--max_h', type=int, default=10, help='maximum height of the bev lifting')
    parser.add_argument('--eval_range', type=float, default=51.2, help='evaluation range for occupancy network')

    parser.add_argument('--grd_h', type=int, default=370, help='ground image height')
    parser.add_argument('--grd_w', type=int, default=1220, help='ground image width')

    parser.add_argument('--head_num', type=int, default=8, help='head number in transformer')
    parser.add_argument('--all_sampling_points', type=int, default=8, help='total sampling points number in cross-attention for one query')
    parser.add_argument('--self_sampling_points', type=int, default=8, help='total sampling points number in self-attention for one query')
    parser.add_argument('--self_level', type=int, default=1, help='number of level of self-attention')
    parser.add_argument('--sat_level', type=int, default=4, help='number of level of sat-feature-pyramid')
    parser.add_argument('--cross_layer_num', type=int, default=6, help='number of layers for cross-attention')
    parser.add_argument('--sat_cross_num', type=int, default=3, help='number of layers for satillite cross-attention')
    parser.add_argument('--self_layer_num', type=int, default=2, help='number of layers for self-attention') 

    parser.add_argument('--num_class', type=int, default=20, help='number of class for prediction')
    args = parser.parse_args()

    return args
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # 使用 PyTorch 设置线程数
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_gpu = torch.cuda.device_count()
    print("device: ", device)
    args = parse_args()
    mini_batch = args.batch_size
    net = Model(args)
    net.to(device)
    net.eval()
    eval(net, args)
