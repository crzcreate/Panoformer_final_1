import pdb
import os

import cv2
import torch
import open3d as op

import numpy as np

def get_file_names(directory):
    file_names = os.listdir(directory)
    return file_names


def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (2*x/float(W)-1.0)*np.pi
    phi = (y/float(H)-0.5)*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),-(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])

def view_pred_val():
    rgb_list   = get_file_names('E:\PanoFormer-main\PanoFormer-main\PanoFormer/val_rgb')
    pred_depth_list   = get_file_names('E:\PanoFormer-main\PanoFormer-main\PanoFormer/pred_depth')
    gt_depth_list     = get_file_names('E:\PanoFormer-main\PanoFormer-main\PanoFormer/gt_depth')
    # layout_depth_list = get_file_names('./layout_depth')

    for rgb, pred_depth, gt_depth in zip(rgb_list, pred_depth_list, gt_depth_list):
        pano_rgb   = cv2.imread('E:\PanoFormer-main\PanoFormer-main\PanoFormer/val_rgb/' + rgb)
        pred_pano_depth   = cv2.imread('E:\PanoFormer-main\PanoFormer-main\PanoFormer/pred_depth/' + pred_depth, -1) / 512
        gt_pano_depth     = cv2.imread('E:\PanoFormer-main\PanoFormer-main\PanoFormer/gt_depth/' + gt_depth, -1) / 512
        # layout_pano_depth = cv2.imread('./layout_depth/' + layout_depth, -1) / 512

        pts3d = get_3Dpoints(pred_pano_depth, 512, 1024)
        print("current dispaly " + rgb + "'s pred pano 3d")
        pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(pts3d.reshape(-1,3))
        pcd.colors = op.utility.Vector3dVector(pano_rgb.reshape(-1,3)/255.)
        op.visualization.draw_geometries([pcd],window_name='pred_pano_depth_room')

        print("current dispaly " + rgb + "'s gt pano 3d")
        gt3ds = get_3Dpoints(gt_pano_depth, 512, 1024)
        #pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(gt3ds.reshape(-1,3))
        pcd.colors = op.utility.Vector3dVector(pano_rgb.reshape(-1,3)/255.)
        op.visualization.draw_geometries([pcd],window_name='gt_pano_depth_room')

        # print("current dispaly " + rgb + "'s layout pano 3d")
        # layout3ds = get_3Dpoints(layout_pano_depth, 512, 1024)
        # pcd.points = op.utility.Vector3dVector(layout3ds.reshape(-1,3))
        # pcd.colors = op.utility.Vector3dVector(pano_rgb.reshape(-1,3)/255.)
        # op.visualization.draw_geometries([pcd],window_name='layout_pano_depth_room')



def view_train_set():
    rgb_list   = []
    depth_list = []
    file_names = get_file_names("../data/train")

    for file_name in file_names:
        if file_name[-7:] == 'rgb.png':
            if file_name[:2] == '._':
                file_name = file_name[2:]
            rgb_list.append(file_name)
            depth_list.append(file_name[:-7] + 'depth.png')



    #pdb.set_trace()

    display_count = 1

    for rgb, depth in zip(rgb_list, depth_list):
        print("current display pano image's name is :", rgb)
        pano_rgb   = cv2.imread('../data/train/' + rgb)
        try:
            pano_depth = cv2.imread('../data/train/' + depth, -1) / 512
        except TypeError:
            pdb.set_trace()

        select_mask= pano_depth > 10
        pano_depth[select_mask] = 10

        ## normalize the depth:
        max_value = pano_depth.max()
        min_value = pano_depth.min()
        normalize = (pano_depth - min_value) / (max_value - min_value)
        normalize = (normalize * 255).astype(np.uint8)


        #pdb.set_trace()

        pts = get_3Dpoints(pano_depth, 2048, 4096)
        #pdb.set_trace()
        pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(pts.reshape(-1,3))
        pcd.colors = op.utility.Vector3dVector(pano_rgb.reshape(-1,3)/255.)

        op.visualization.draw_geometries([pcd],window_name='pano_depth_room')

        save_flag = input("whether save current normalize file?")
        if save_flag == 'yes':
            path = "/home/emiyaning/Pictures/display_sence/sence" + str(display_count) + "/" + depth
            display_count = display_count + 1
            cv2.imwrite(path, normalize)
            print("current save the path: ", path)


if __name__ == '__main__':
    #view_train_set()
    view_pred_val()

    #pdb.set_trace()