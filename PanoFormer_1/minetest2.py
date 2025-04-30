import torch
pth1="E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\\panodepth\models\\test.pth"
pth2="E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\\panodepth\models\\best_epoch_222.pth"
state_dict1=torch.load(pth1)
state_dict2=(torch.load(pth2))["state_dict"]

print(state_dict1)