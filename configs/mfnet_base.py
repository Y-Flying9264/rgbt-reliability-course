import os
import os.path as osp
import sys
import time
import numpy as np
import argparse

from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

# —— 开启 FRM+ 总开关 ——
C.use_frm_plus = True

# CM-FRM置信度 Settings
C.frm_plus = edict()
C.frm_plus.use_conf = True                   # False: 关闭像素置信度
C.frm_plus.use_learnable_lambda = True       # False: 用常数 λ_c/λ_s（见下两项）
C.frm_plus.lambda_c_const = 0.5              # Only-conf/Baseline 时用到
C.frm_plus.lambda_s_const = 0.5
C.frm_plus.lambda_init = 0.5                 # 学习 λ 的初始化目标（靠近原 0.5）
C.frm_plus.conf_init = 0.95                  # 置信度初始化（偏高，避免初期抑制过强）
C.frm_plus.cgate_reduction = 8               # 通道门 MLP 降维倍率
C.frm_plus.sgate_reduction = 4               # 空间门降维倍率
C.frm_plus.conf_reduction = 4                # 置信度头降维倍率

C.seed = 12345

remoteip = os.getcwd()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'MFNet'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'MFNet')
C.rgb_root_folder = osp.join(C.dataset_path, 'images')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'labels')
C.gt_format = '.png'
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Thermal')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "val.txt")


C.is_test = False

C.num_classes = 9
C.class_names = [
    'background', 'car', 'person', 'bike', 'curve',
    'car_stop', 'guardrail', 'color_cone', 'bump']

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'swiftformer_s' # Remember change the path below.
C.pretrained_model = C.root_dir + '/pretrained/swiftformer/SwiftFormer_S.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

# ---- Attention Gate (AG) 开关（默认关闭，保证可回退）----
C.use_ag = True           # True 开启 AG、False 关闭
C.ag_mid = 32             # 16~64 均可，32 较稳
C.ag_use_dwconv = True    # 使用 DWConv 增强局部建模
C.ag_stages = (1, 1, 1)   # (C1<-C2, C2<-C3, C3<-C4)


"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 400

C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

# --- auto infer dataset sizes from txt ---
def _count_lines(p):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

C.num_train_imgs = _count_lines(C.train_source)
C.num_eval_imgs  = _count_lines(C.eval_source)
C.niters_per_epoch = max(1, C.num_train_imgs // C.batch_size + 1)


C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()