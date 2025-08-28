# This is a sample Python script.
import argparse
import pdb
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ExponentialLR
from kitti_dataset import *
from model.models import Model
from utils import *
from model.evaluation import Evaluation
from torch.utils.tensorboard import SummaryWriter
from model.utils.ssc_metric import SSCMetrics
import tracemalloc
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from thop import profile


from torch.profiler import profiler, record_function, ProfilerActivity

n_classes = 20
class_names = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                    "motorcyclist", "road",
                    "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                    "terrain", "pole", "traffic-sign", ]
metrics = SSCMetrics(20)


def ddp_setup(rank, world_size):
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch, step, len_epoch, initial_lr):
    current_iter = epoch * len_epoch + step
    warmup_iters = 500
    warmup_ratio = 1.0 / 3

    if current_iter < warmup_iters:
        lr = initial_lr * warmup_ratio + step / warmup_iters * (initial_lr - initial_lr * warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def eval(net_eval, args, epoch_num, writer=None, world_size=None, device="cuda:0"):
    mini_batch = args.batch_size
    eval_loader = load_eval_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range, args = args)
    net_eval.load_state_dict(torch.load(f"check_point/semanticKITTI.pth"))
    results = []
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
    if device == 0 or world_size == None:
        print(metrics.count)
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
        # writer.add_text('mIou', str(detail), global_step=epoch_num)
    return

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {torch.mean(param.grad)}")

def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()

def train(net, lr, args, world_size=None, rank=None):
    print("begin training! ")
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr * 1e-3)
    # scheduler = ExponentialLR(optimizer, gamma, last_epoch=-1,verbose=False)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=(3673*20) + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos'
    )

    epoch_count = 0
    mini_batch = args.batch_size
    # if rank==0 or world_size==None:
    #     writer_dir = "./log_dir"
    #     writer = SummaryWriter(writer_dir)
    #     writer.add_text('settings', "baseline method", global_step=epoch_count)

    for epoch in range(args.epochs):
        net.train()
        "Set batchnorm into eval mode"
        if world_size == None:
            net.image_backbone.apply(set_bn_eval)
            # Set layer1 to eval mode
            # net.image_backbone.layer1.eval()
            "Set batchnorm into eval mode"
            net.sat_backbone.apply(set_bn_eval)
            # Set layer1 to eval mode
            net.sat_backbone.layer1.eval()
        else:
            net.module.image_backbone.apply(set_bn_eval)
            # Set layer1 to eval mode
            # net.image_backbone.layer1.eval()
            "Set batchnorm into eval mode"
            net.module.sat_backbone.apply(set_bn_eval)
            # Set layer1 to eval mode
            net.module.sat_backbone.layer1.eval()

        optimizer.zero_grad()
        trainloader = load_train_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range, args = args, world_size=world_size, rank=rank)
        count = 0
        if rank == 0 or world_size == None:
            print(f"===============================epoch{epoch_count}==================================")
        # for Loop, Data in enumerate(trainloader, 0):
        #     adjust_learning_rate(optimizer, epoch, Loop, len(trainloader), lr)
        #     optimizer.zero_grad()
        #     gd_image, sat_map, image_meta, image_list, gt = [item for item in Data[:-1]]
        #     if world_size == None:
        #         gd_image = gd_image.to(device)
        #         sat_map = sat_map.to(device)
        #         gt = gt.to(device)
        #     else:
        #         gd_image = gd_image.to(rank)
        #         sat_map = sat_map.to(rank)
        #         gt = gt.to(rank)
        #     "=================================bev training=============================================="
        #     bev_gt, bev_prob = bev_compress(gt, args.num_class)
        #     bev_gt = bev_gt.unsqueeze(0).type_as(gt)
        #     bev_prob = bev_prob.unsqueeze(0).type_as(gt)
        #     "==============================================================================================="
        #     loss_dict = net(gd_image, sat_map, image_meta, gt, bev_gt, bev_prob, "train")
        #     # memory = torch.cuda.memory_allocated()
        #     # print(memory)
        #     loss_ssc = loss_dict["loss_ssc"]
        #     loss_bev = loss_dict["loss_bev"]
        #     loss_sem = loss_dict["loss_sem_scal"]
        #     loss_geo = loss_dict["loss_geo_scal"]
        #     loss = loss_ssc  + loss_sem  + loss_geo  + loss_bev
        #     torch.cuda.empty_cache()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
        #     # grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
        #     optimizer.step()  # This step is responsible for updating weights
        #     optimizer.zero_grad()
        #     if rank==0 or world_size==None:
        #         print(f"\repoch: {epoch_count}, data: {count} / {len(trainloader)}, loss_ssc: {loss_ssc}, loss_sem_scal: {loss_sem}, geo_loss: {loss_geo}",end="")
        #     count = count + 1
        #     scheduler.step()
        if world_size==None:
            print(f"\nend epoch training, begin eval: ")
            with torch.no_grad():
                net.eval()
                eval(net, args, epoch_count)
            epoch_count += 1
        elif rank==0:
            print(f"\nend epoch training, begin eval: ")

            with torch.no_grad():
                net.eval()
                eval(net.module, args, epoch_count, world_size=world_size, device=rank)
            epoch_count += 1

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
def process_train(rank, world_size, args):
    ddp_setup(rank, world_size)
    set_random_seed(1, True)
    net = Model(args)
    net.to(rank)
    net = DDP(net, device_ids=[rank])
    lr = args.lr
    train(net, lr, args, world_size=world_size, rank=rank)


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
    lr = args.lr
    if num_gpu > 1:
        os.environ["MASTER_ADDR"] = "localhost"# ——11——
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(process_train, nprocs=num_gpu, args=(num_gpu, args), join=True)
    else:
        net = Model(args)
        set_random_seed(1, True)
        net.to(device)
        train(net, lr, args)
