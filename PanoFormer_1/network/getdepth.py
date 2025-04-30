import sys
from shapely.geometry import Polygon
from scipy.ndimage.filters import maximum_filter
import torch
from network import post_proc, panostretch
import numpy as np

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]
def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys
def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth,hf的输入其实是一个偏差量，需要relu，有可能是负也有可能是正的，这样能保证输出的最终floor_h是正常的
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]
    #这里的判断标准是得到的depth是否存在0
    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth
#这个版本是h=1.6默认值的情况
# def layout_2_depth(cor_id, h, w, return_mask=False):
#     # Convert corners to per-column boundary first
#     # Up -pi/2,  Down pi/2
#     vc, vf = cor_2_1d(cor_id, h, w)
#     vc = vc[None, :]  # [1, w]
#     vf = vf[None, :]  # [1, w]
#     assert (vc > 0).sum() == 0
#     assert (vf < 0).sum() == 0
#
#     # Per-pixel v coordinate (vertical angle)
#     vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
#     vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]
#
#     # Floor-plane to depth这里可以改成1.5，理论上会让指标更好，现在尝试1.43,1.56(所有深度图的均值)
#     floor_h = 1.6
#     floor_d = np.abs(floor_h / np.sin(vs))
#
#     # wall to camera distance on horizontal plane at cross camera center
#     cs = floor_h / np.tan(vf)
#
#     # Ceiling-plane to depth
#     ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
#     ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]
#
#     # Wall to depth
#     wall_d = np.abs(cs / np.cos(vs))  # [h, w]
#
#     # Recover layout depth
#     floor_mask = (vs > vf)
#     ceil_mask = (vs < vc)
#     wall_mask = (~floor_mask) & (~ceil_mask)
#     depth = np.zeros([h, w], np.float32)    # [h, w]
#     depth[floor_mask] = floor_d[floor_mask]
#     depth[ceil_mask] = ceil_d[ceil_mask]
#     depth[wall_mask] = wall_d[wall_mask]
#
#     assert (depth == 0).sum() == 0
#     if return_mask:
#         return depth, floor_mask, ceil_mask, wall_mask
#     return depth
# def get_h(y_bon,depth,H,W):
#     H, W = tuple((512, 1024))
#     # y_cor_ = torch.sigmoid(y_cor_.detach().cpu())
#     # y_cor_ = np.array(y_cor_)
#     # depth = depth.detach().cpu()
#     # depth = np.array(depth)
#     # batch = depth.shape[0]
#     # depth_list = []
#     y1 = np.round(y_bon[0]).astype(int)
#     y2 = np.round(y_bon[1]).astype(int)
#     y2_idx=[(h,w)for h,w in zip(y2,np.arange(W))]
#     # y2_idx=[np.arange(W),y2]
#     depth=depth[0]
#     depth_floor=[depth[h][w] for (h,w) in y2_idx]
#     # depth_floor=[depth[h][w] for (h, w) in y1_idx]
#     y_floor=((y1-0.5)/H-0.5)*np.pi
#     hf=np.abs(depth_floor*np.sin(y_floor))
#     hf=hf.mean()
#     # if hf==0:
#     #     print(hf)
#     return hf
# def inference(y_bon_,y_cor_,hf_,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
#     '''
#     net   : the trained HorizonNet
#     x     : tensor in shape [1, 3, 512, 1024]
#     flip  : fliping testing augmentation
#     rotate: horizontal rotation testing augmentation
#     '''
#     H, W = tuple((512,1024))
#     batch = y_bon_.shape[0]
#     y_bon_=y_bon_.detach().cpu()
#     y_bon_=np.array(y_bon_)
#     y_cor_=torch.sigmoid(y_cor_.detach().cpu())
#     y_cor_ = np.array(y_cor_)
#     hf_=hf_.detach().cpu()
#     depth_list=[]
#     #将输出的每个y_bon的点直接去算他对应的点的y坐标，转化到512范围，然后再找到输入的深度图对应的位置的depth，然后得到对应的h1,h2
#     for b in range(batch):
#         y_bon = (y_bon_[b] / np.pi + 0.5) * H - 0.5
#         y_bon[0] = np.clip(y_bon[0], 1, H/2-1)
#         y_bon[1] = np.clip(y_bon[1], H/2+1, H-2)
#         y_cor = y_cor_[b, 0]
#         hf=hf_[b]
#         # Init floor/ceil plane 这个是将底部的高度统一化为z_mean这里是为了得到cor_id是2048×2的输出
#         #这里必须重新定义一下，不然的话会反复利用出现循环bug
#         z0 = 50
#         r = 0.05
#         min_v=None
#         _, z1 = post_proc.np_refine_by_fix_z(*y_bon, z0)
#         if min_v is None:
#             min_v = 0 if force_cuboid else 0.05
#         r = int(round(W * r / 2))
#         N = 4 if force_cuboid else None
#         xs_ = find_N_peaks(y_cor, r=r, min_v=min_v, N=N)[0]
#         # Generate wall-walls
#         cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
#         if not force_cuboid:
#             # Check valid (for fear self-intersection)
#             xy2d = np.zeros((len(xy_cor), 2), np.float32)
#             for i in range(len(xy_cor)):
#                 xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
#                 xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
#             # try:
#             #     Polygon(xy2d)
#             # except ValueError as e:
#             #     #如果创建多边形时发生 ValueError 异常，打印错误信息
#             #     print(f"创建多边形失败：{e}")
#             if xy2d.shape[0]<=2:
#                 xs_ = find_N_peaks(y_cor, r=r, min_v=0, N=4)[0]
#                 cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)
#             elif not Polygon(xy2d).is_valid:
#                 # print(
#                 #     'Fail to generate valid general layout!! '
#                 #     'Generate cuboid as fallback.')
#                 xs_ = find_N_peaks(y_cor, r=r, min_v=0, N=4)[0]
#                 cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)
#         cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
#         # Collect corner position in equirectangular
#         cor_id = np.zeros((len(cor)*2, 2), np.float32)
#         for j in range(len(cor)):
#             cor_id[j*2] = cor[j, 0], cor[j, 1]
#             cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
#         # 不能Normalized to [0, 1]要保留到512×1024
#         #现在要根据这个cor_id还有根据layout得到的边界，去得到上下边界的h1和h2，然后导入layout_2_depth
#         depth_out = layout_2_depth(cor_id,hf, H, W, return_mask=False)
#         depth_list.append(depth_out)
#     depth=np.stack(depth_list,axis=0)
#     return torch.from_numpy(depth)






# def inference(y_bon_,y_cor_,depth,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
#     '''
#     net   : the trained HorizonNet
#     x     : tensor in shape [1, 3, 512, 1024]
#     flip  : fliping testing augmentation
#     rotate: horizontal rotation testing augmentation
#     '''
#     H, W = tuple((512,1024))
#     y_bon_=y_bon_.detach().cpu()
#     y_bon_=np.array(y_bon_)
#     y_cor_=torch.sigmoid(y_cor_.detach().cpu())
#     y_cor_ = np.array(y_cor_)
#     depth=depth.detach().cpu()
#     depth=np.array(depth)
#     batch=depth.shape[0]
#     depth_list=[]
#     #将输出的每个y_bon的点直接去算他对应的点的y坐标，转化到512范围，然后再找到输入的深度图对应的位置的depth，然后得到对应的h1,h2
#     for b in range(batch):
#         y_bon = (y_bon_[b] / np.pi + 0.5) * H - 0.5
#         y_bon[0] = np.clip(y_bon[0], 1, H/2-1)
#         y_bon[1] = np.clip(y_bon[1], H/2+1, H-2)
#         y_cor = y_cor_[b, 0]
#         depthb=depth[b]
#         #保持疑问
#         # hf=get_h(y_bon,depthb,H,W)
#         # Init floor/ceil plane 这个是将底部的高度统一化为z_mean这里是为了得到cor_id是2048×2的输出
#         #这里必须重新定义一下，不然的话会反复利用出现循环bug
#         z0 = 50
#         r = 0.05
#         min_v=None
#         _, z1 = post_proc.np_refine_by_fix_z(*y_bon, z0)
#         if min_v is None:
#             min_v = 0 if force_cuboid else 0.05
#         r = int(round(W * r / 2))
#         N = 4 if force_cuboid else None
#         xs_ = find_N_peaks(y_cor, r=r, min_v=min_v, N=N)[0]
#         # Generate wall-walls
#         cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
#         if not force_cuboid:
#             # Check valid (for fear self-intersection)
#             xy2d = np.zeros((len(xy_cor), 2), np.float32)
#             for i in range(len(xy_cor)):
#                 xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
#                 xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
#             # try:
#             #     Polygon(xy2d)
#             # except ValueError as e:
#             #     #如果创建多边形时发生 ValueError 异常，打印错误信息
#             #     print(f"创建多边形失败：{e}")
#             if xy2d.shape[0]<=2:
#                 xs_ = find_N_peaks(y_cor, r=r, min_v=0, N=4)[0]
#                 cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)
#             elif not Polygon(xy2d).is_valid:
#                 # print(
#                 #     'Fail to generate valid general layout!! '
#                 #     'Generate cuboid as fallback.')
#                 xs_ = find_N_peaks(y_cor, r=r, min_v=0, N=4)[0]
#                 cor, xy_cor = post_proc.gen_ww(xs_, y_bon[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)
#         cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
#         # Collect corner position in equirectangular
#         cor_id = np.zeros((len(cor)*2, 2), np.float32)
#         for j in range(len(cor)):
#             cor_id[j*2] = cor[j, 0], cor[j, 1]
#             cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
#         # 不能Normalized to [0, 1]要保留到512×1024
#         #现在要根据这个cor_id还有根据layout得到的边界，去得到上下边界的h1和h2，然后导入layout_2_depth
#         depth_out = layout_2_depth(cor_id,hf, H, W, return_mask=False)
#         depth_list.append(depth_out)
#     depth=np.stack(depth_list,axis=0)
#     return torch.from_numpy(depth)
def inference(y_bon_,y_cor_,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''
    H, W = tuple((512,1024))
    y_bon_=y_bon_.detach().cpu()
    y_bon_=np.array(y_bon_)
    y_cor_=torch.sigmoid(y_cor_.detach().cpu())
    y_cor_ = np.array(y_cor_)
    batch=y_bon_.shape[0]
    depth_list=[]
    #将输出的每个y_bon的点直接去算他对应的点的y坐标，转化到512范围，然后再找到输入的深度图对应的位置的depth，然后得到对应的h1,h2
    for i in range(batch):
        y_bon = (y_bon_[i] / np.pi + 0.5) * H - 0.5
        y_bon[0] = np.clip(y_bon[0], 1, H/2-1)
        y_bon[1] = np.clip(y_bon[1], H/2+1, H-2)
        # Init floor/ceil plane 这个是将底部的高度统一化为z_mean这里是为了得到cor_id是2048×2的输出
        z0 = 50
        _, z1 = post_proc.np_refine_by_fix_z(*y_bon, z0)
        cor = np.stack([np.arange(1024), y_bon[0]], 1)
        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor)*2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j*2] = cor[j, 0], cor[j, 1]
            cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
        # 不能Normalized to [0, 1]要保留到512×1024
        #现在要根据这个cor_id还有根据layout得到的边界，然后导入layout_2_depth
        depth_out = layout_2_depth(cor_id, H, W, return_mask=False)
        depth_out=depth_out+0.5
        #depth+1
        #depth*1.1+0.8
        #depth*1.5 可能会更好
        #这里要加多少的余量才能保证墙体学的很好，得测试一下1.depth+1,然后depth1的输出变成gelu，这样会不会好点,目前看来可能加1不太行，乘以1.2说不定可以
        depth_list.append(depth_out)
    depth=np.stack(depth_list,axis=0)
    return torch.from_numpy(depth)

#这个版本是用layout得到深度图然后默认h为1.6得到的
# def inference(y_bon_,y_cor_,depth,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
#     '''
#     net   : the trained HorizonNet
#     x     : tensor in shape [1, 3, 512, 1024]
#     flip  : fliping testing augmentation
#     rotate: horizontal rotation testing augmentation
#     '''
#     H, W = tuple((512,1024))
#     y_bon_=y_bon_.detach().cpu()
#     y_bon_=np.array(y_bon_)
#     y_cor_=torch.sigmoid(y_cor_.detach().cpu())
#     y_cor_ = np.array(y_cor_)
#     depth=depth.detach().cpu()
#     depth=np.array(depth)
#     batch=depth.shape[0]
#     depth_list=[]
#     #将输出的每个y_bon的点直接去算他对应的点的y坐标，转化到512范围，然后再找到输入的深度图对应的位置的depth，然后得到对应的h1,h2
#     for i in range(batch):
#         y_bon = (y_bon_[i] / np.pi + 0.5) * H - 0.5
#         y_bon[0] = np.clip(y_bon[0], 1, H/2-1)
#         y_bon[1] = np.clip(y_bon[1], H/2+1, H-2)
#         # Init floor/ceil plane 这个是将底部的高度统一化为z_mean
#         z0 = 50
#         _, z1 = post_proc.np_refine_by_fix_z(*y_bon, z0)
#         cor = np.stack([np.arange(1024), y_bon[0]], 1)
#         cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
#         # Collect corner position in equirectangular
#         cor_id = np.zeros((len(cor)*2, 2), np.float32)
#         for j in range(len(cor)):
#             cor_id[j*2] = cor[j, 0], cor[j, 1]
#             cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
#         # 不能Normalized to [0, 1]要保留到512×1024
#         depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)
#         depth_list.append(depth)
#     depth=np.stack(depth_list,axis=0)
#     return torch.from_numpy(depth)

# def inference(y_bon_,y_cor_,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
#     '''
#     net   : the trained HorizonNet
#     x     : tensor in shape [1, 3, 512, 1024]
#     flip  : fliping testing augmentation
#     rotate: horizontal rotation testing augmentation
#     '''
#     H, W = tuple((512,1024))
#     y_bon_=y_bon_.detach().cpu()
#     y_bon_=np.array(y_bon_)
#     y_cor_=torch.sigmoid(y_cor_.detach().cpu())
#     y_cor_ = np.array(y_cor_)
#     # Network feedforward (with testing augmentation)
#     y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
#     y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
#     y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
#     y_cor_ = y_cor_[0, 0]
#     # Init floor/ceil plane 这个是将底部的高度统一化为z_mean
#     z0 = 50
#     _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)
#     cor = np.stack([np.arange(1024), y_bon_[0]], 1)
#     cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
#     # Collect corner position in equirectangular
#     cor_id = np.zeros((len(cor)*2, 2), np.float32)
#     for j in range(len(cor)):
#         cor_id[j*2] = cor[j, 0], cor[j, 1]
#         cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
#     # 不能Normalized to [0, 1]要保留到512×1024
#     depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)
#     return torch.from_numpy(depth)