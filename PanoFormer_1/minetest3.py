import cv2
import numpy as np
depth_name="E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models\camera_01d25d1241b14f61bd143df9c758b914_hallway_14_frame_equirectangular_domain_depth.png"

max_depth_meters=10*512
gt_depth = cv2.imread(depth_name,-1)
gt_depth = cv2.resize(gt_depth, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
gt_type=type(gt_depth[0][0])
# gt_depth = gt_depth.astype(np.float)/512
gt_depth[gt_depth > max_depth_meters+1] = max_depth_meters + 1
cv2.imwrite("E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models\hallway_14_frame_equirectangular_domain_depth.png",gt_depth)