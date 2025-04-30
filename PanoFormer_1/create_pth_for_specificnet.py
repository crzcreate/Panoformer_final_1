import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import argparse
import logging
import os
import os.path as osp

import re
from collections import OrderedDict
import torch

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a 3D detector')
#     parser.add_argument('--config', default="E:/mmdetection3d/configs/pointnet2/pointnet2_msg_panorama_test.py",help='train config file path')
#     parser.add_argument('--work_dir',help='the dir to save logs and models')
#     # print(parser.config)
#     # input()
#     parser.add_argument(
#         '--amp',
#         action='store_true',
#         default=False,
#         help='enable automatic-mixed-precision training')
#     parser.add_argument(
#         '--sync_bn',
#         choices=['none', 'torch', 'mmcv'],
#         default='none',
#         help='convert all BatchNorm layers in the model to SyncBatchNorm '
#         '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
#     parser.add_argument(
#         '--auto-scale-lr',
#         action='store_true',
#         help='enable automatically scaling LR.')
#     parser.add_argument(
#         '--resume',
#         default="E:/mmdetection3d/tools/work_dirs/resnet50_rnn__mp3d.pth",
#         nargs='?',
#         type=str,
#         const='auto',
#         help='If specify checkpoint path, resume from it, while if not '
#         'specify, try to auto resume from the latest checkpoint '
#         'in the work directory.')
#     parser.add_argument(
#         '--ceph', action='store_true', help='Use ceph as data storage backend')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
#     # will pass the `--local-rank` parameter to `tools/train.py` instead
#     # of `--local_rank`.
#     parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#     return args
# args = parse_args()
# # print(args.config)
# # load config
# cfg = Config.fromfile(args.config)
# # print(cfg)
# # TODO: We will unify the ceph support approach with other OpenMMLab repos
# if args.ceph:
#     cfg = replace_ceph_backend(cfg)
# cfg.launcher = args.launcher
# if args.cfg_options is not None:
#     cfg.merge_from_dict(args.cfg_options)
# # work_dir is determined in this priority: CLI > segment in file > filename
# if args.work_dir is not None:
#     # update configs according to CLI args if args.work_dir is not None
#     cfg.work_dir = args.work_dir
# elif cfg.get('work_dir', None) is None:
#     # use config filename as default work_dir if cfg.work_dir is None
#     cfg.work_dir = osp.join('./work_dirs',
#                             osp.splitext(osp.basename(args.config))[0])
#     # print(cfg.work_dir)
# # enable automatic-mixed-precision training
# if args.amp is True:
#     optim_wrapper = cfg.optim_wrapper.type
#     if optim_wrapper == 'AmpOptimWrapper':
#         print_log(
#             'AMP training is already enabled in your config.',
#             logger='current',
#             level=logging.WARNING)
#     else:
#         assert optim_wrapper == 'OptimWrapper', (
#             '`--amp` is only supported when the optimizer wrapper type is '
#             f'`OptimWrapper` but got {optim_wrapper}.')
#         cfg.optim_wrapper.type = 'AmpOptimWrapper'
#         cfg.optim_wrapper.loss_scale = 'dynamic'
# # convert BatchNorm layers
# if args.sync_bn != 'none':
#     cfg.sync_bn = args.sync_bn
# # enable automatically scaling LR
# if args.auto_scale_lr:
#     if 'auto_scale_lr' in cfg and \
#             'enable' in cfg.auto_scale_lr and \
#             'base_batch_size' in cfg.auto_scale_lr:
#         cfg.auto_scale_lr.enable = True
#     else:
#         raise RuntimeError('Can not find "auto_scale_lr" or '
#                            '"auto_scale_lr.enable" or '
#                            '"auto_scale_lr.base_batch_size" in your'
#                            ' configuration file.')
#
# # resume is determined in this priority: resume from > auto_resume
# if args.resume == 'auto':
#     cfg.resume = None
#     cfg.load_from = args.resume
# elif args.resume is not None:
#     cfg.resume = False
#     cfg.load_from = args.resume
#
# # build the runner from config
# if 'runner_type' not in cfg:
#     # build the default runner
#     runner = Runner.from_cfg(cfg)
# else:
#     # build customized runner from the registry
#     # if 'runner_type' is set in the cfg
#     runner = RUNNERS.build(cfg)

# # horizonnet=runner.model
# # super_ckpt=horizonnet.state_dict()
# # #E:\HorizonNet-master\\resnet50_rnn__mp3d.pth
# # if 'state_dict' in ckpt:
# #     _state_dict = ckpt['state_dict']
# # elif 'model' in ckpt:
# #     _state_dict = ckpt['model']
# # else:
# #     _state_dict = ckpt
check_point="E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models\\best_epoch_42.pth"
ckpt = torch.load(check_point, map_location='cpu')
ckpt=ckpt["state_dict"]
new_state_dict=OrderedDict()
# 定义修订规则
for key,value in ckpt.items():
    if "feature_extractor" in key:
        key=key.replace("feature_extractor.","")
        # if "output_proj_" in key and ".0.1" in key:
        #     key=key.replace(".0.1",".0")
        #     new_state_dict[key]=value
        # else:
        new_state_dict[key]=value
    else:
        key="lstmdecoder."+key
        new_state_dict[key]=value
torch.save(new_state_dict, "E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models\\test.pth")

print("字典已保存为 my_dict.pth 文件")

# 修改键名
# check_point="E:\mmdetection3d/tools\work_dirs\\panoformer.pth"
# ckpt = torch.load(check_point, map_location='cpu')
# new_state_dict=OrderedDict()
# # 定义修订规则
# for key,value in ckpt.items():
#         key="backbone."+key
#         new_state_dict[key]=value
# torch.save(new_state_dict, "E:/mmdetection3d/tools/work_dirs/panoformer_backbone.pth")
#
# print("字典已保存为 my_dict.pth 文件")



#平接两个state_dict

# pano_dict=torch.load("E:/mmdetection3d/tools/work_dirs/panoformer_backbone.pth")
# lstm_dict=torch.load("E:/mmdetection3d/tools/work_dirs/lstm.pth")
#
# new_state_dict=OrderedDict()
# for key,value in pano_dict.items():
#     new_state_dict[key]=value
# for key,value in lstm_dict.items():
#     new_state_dict[key]=value
# torch.save(new_state_dict, "E:/mmdetection3d/tools/work_dirs/pano_lstm.pth")

