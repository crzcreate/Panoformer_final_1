import os
import cv2
import torch
import argparse
import numpy as np
import open3d as op
from tqdm import tqdm
import torch.nn as nn
# def get_3Dpoints(dep,H,W):
#     x,y = np.meshgrid(np.arange(W),np.arange(H))
#     theta = (1.0-2*x/float(W))*np.pi
#     phi = (0.5-y/float(H))*np.pi
#     ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
#     vec = np.array([(cp*ct),(cp*st),(sp)])
#     pts = vec * dep
#     distance_threshold = 10000
#     # 计算每个点到原点的距离
#     # 将超过阈值的点设置为 [0, 0, 0]
#     pts=pts.transpose(1,2,0)
#     for i in range(pts.shape[0]):
#         for j in range(pts.shape[1]):
#             distance = np.sqrt(sum([x**2 for x in pts[i][j]]))
#             if distance>distance_threshold:
#                 pts[i][j]=[0,0,0]
#     return pts
def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (1.0-2*x/float(W))*np.pi
    phi = (0.5-y/float(H))*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])

    #Output resolution

def get_voxel_from_depth(depth,img):
    #x 是n1hw的深度图，里面的值就是正常情况下的米数，然后要给他变成体素，每个点其实就是体素，不过可以看做点云
    H,W=512,1024
    dep=depth.numpy()
    pts=get_3Dpoints(dep,H,W)
    pcd = op.geometry.PointCloud()
    pcd.points = op.utility.Vector3dVector(pts.reshape(-1, 3))
    pcd.colors = op.utility.Vector3dVector(img.reshape(-1, img.shape[1]) / 255.)
    return  pcd
# cv2.imshow(name, seg)
# cv2.waitKey(0)
def get_voxel(feature_list):
    voxel_list=[]
    for feature in feature_list:
        feature=np.array(feature.cpu())#N 1+(256,512,1024,2048) ,hw(128.256 64.128  32.64  16.32)
        depth=feature[:,0,:,:]
        img=feature[:,1:,:,:]
        voxel=get_voxel_from_depth(depth,img)
        voxel_list.append(voxel)
    return voxel_list