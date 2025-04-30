import os
import torch.nn as nn
import numpy
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

max_depth_meters=10
depth_image_dir="E:\PanoFormer-main\PanoFormer-main\PanoFormer\data\panotodepth\\train"
depth_list=[]
depth_list_higher=[]
depth_list_lower=[]
for i in os.listdir(depth_image_dir):
    if "depth" in i:
        gt_depth = cv2.imread(os.path.join(depth_image_dir,i), -1)
        gt_depth = cv2.resize(gt_depth, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float) / 512
        gt_depth[gt_depth > max_depth_meters + 1] = max_depth_meters + 1
        gt_depth_mean=gt_depth[:5,:].mean()
        if gt_depth_mean<=2.0 and gt_depth_mean>=1.1:
            depth_list.append(gt_depth[:5,:].mean())
        elif gt_depth_mean>2.0:
            depth_list_higher.append(gt_depth_mean)
        else:
            depth_list_lower.append(gt_depth_mean)
depth_mean=sum(depth_list)/len(depth_list)
print(depth_mean)

