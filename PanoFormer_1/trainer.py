from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from PIL import Image
import cv2
torch.manual_seed(594277)
torch.cuda.manual_seed(594277)
from metrics import compute_depth_metrics, Evaluator
from losses import BerhuLoss
import loss_gradient as loss_g
from network.model import Panoformer as PanoBiT
from stanford2d3d import Stanford2D3D
from matterport3d import Matterport3D
from visualize import show_depth

def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y
def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
    y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
    y_cor = y_cor.numpy()

    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy() * 0.5).astype(np.uint8)
    y1 = np.round(y_bon[0]).astype(int)
    y2 = np.round(y_bon[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

    return np.concatenate([gt_cor, img_pad, img_bon], 0)

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        train_dataset = Matterport3D('E:\HorizonNet-master\orignal_panotodepth\panotodepth/train/','./rgb_depth_train.txt', self.settings.disable_color_augmentation, self.settings.disable_LR_filp_augmentation,
                                     self.settings.disable_yaw_rotation_augmentation, is_training=True)#E:/PanoFormer-main/PanoFormer-main/PanoFormer/data/panotodepth/train/

        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs
        val_dataset = Matterport3D('E:\HorizonNet-master\orignal_panotodepth\panotodepth/val/','./rgb_depth_val.txt', self.settings.disable_color_augmentation, self.settings.disable_LR_filp_augmentation,
                                    self.settings.disable_yaw_rotation_augmentation, is_training=False)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
                                     
        self.model = PanoBiT()

        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self.output_file="E:/PanoFormer-main/PanoFormer-main/PanoFormer/tmp/panodepth/models/losses.txt"
        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)
            # pred_depth=outputs["pred_depth"]
            # for i in range(pred_depth.shape[0]):
            #     show_depth(pred_depth[i].squeeze(0))
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                
                pred_depth = outputs["pred_depth"].detach() * mask
                gt_depth = inputs["gt_depth"] * mask
                

                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                self.log("train", inputs, outputs, losses)

            self.step += 1

    # def _prepare_x(self, x):
    #     if self.x_mean.device != x.device:
    #         self.x_mean = self.x_mean.to(x.device)
    #         self.x_std = self.x_std.to(x.device)
    #     return (x[:, :3] - self.x_mean) / self.x_std
    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb"]:#这个name要删了
                inputs[key] = ipt.to(self.device)

        losses = {}
        #print(inputs["val_mask"].size())

        equi_inputs = inputs["normalized_rgb"]# * inputs["val_mask"]

        # cube_inputs = inputs["normalized_cube_rgb"]

        outputs = self.model(equi_inputs)

        gt = inputs["gt_depth"] * inputs["val_mask"]
        pred = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        #输出一下中间部分的depth，然后可视化一下
        # 保存为 PNG 文件
        # name=inputs["name"]
        # for i,i_name in zip(range(len(name)),name):
        #     depth=np.array((pred[i][0] * 512).cpu())
        #     depth=np.uint16(depth)
        #     cv2.imwrite(os.path.join("E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models",f"depth_{i_name}"),depth )
        # # #用bon和cor去测试输出图
        # for i,i_name in zip(range(len(name)),name):
        #     vis_outi = cv2.cvtColor(visualize_a_data(equi_inputs[i].cpu(), outputs["bon"][i].cpu(), outputs["cor"][i].cpu()),cv2.COLOR_BGR2RGB)
        #     vis_path = f"E:\PanoFormer-main\PanoFormer-main\PanoFormer\\tmp\panodepth\models\layout_{i_name}.png"
        #     Image.fromarray(vis_outi).save(vis_path)

        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(pred)
        #dmap = get_dmap(self.settings.batch_size)
        losses["loss"] = self.compute_loss(inputs["gt_depth"].float
                                            () * inputs["val_mask"], outputs["pred_depth"]) +\
                         self.compute_loss(G_x, p_x) +\
                         self.compute_loss(G_y, p_y)

        #outputs["pred_depth"] = pred[inputs["val_mask"]]


        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))
        total_loss=0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs)
                total_loss+=losses["loss"]
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]
                #输出gt_depth,layout的深度，还有最终的深度

                # import pdb
                # pdb.set_trace()这之后是输出可视化depth和rgb的
                # pred_depth1 = (pred_depth[0] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                # pred_depth2 = (pred_depth[1] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                #
                # # layout_depth1 = (layout_depth[0] * 512).permute(1, 2, 0).cpu().numpy().astype(np.uint16)
                # # layout_depth2 = (layout_depth[1] * 512).permute(1, 2, 0).cpu().numpy().astype(np.uint16)
                #
                # gt_depth1 = (gt_depth[0] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                # gt_depth2 = (gt_depth[1] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                #
                # # item_rate1= (outputs['item_rate'][0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
                # # item_rate2= (outputs['item_rate'][1].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
                #
                # rgb1 = (inputs["rgb"][0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
                # rgb2 = (inputs['rgb'][1].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
                #
                # cv2.imwrite('./val_rgb/' + str(batch_idx * 2 + 0) + ".png", rgb1)
                # cv2.imwrite('./val_rgb/' + str(batch_idx * 2 + 1) + ".png", rgb2)
                #
                # # cv2.imwrite("./layout_depth/" + str(batch_idx * 2 + 0) + ".png", layout_depth1)
                # # cv2.imwrite("./layout_depth/" + str(batch_idx * 2 + 1) + ".png", layout_depth2)
                #
                # cv2.imwrite('./pred_depth/' + str(batch_idx * 2 + 0) + ".png", pred_depth1)
                # cv2.imwrite('./pred_depth/' + str(batch_idx * 2 + 1) + ".png", pred_depth2)
                #
                # cv2.imwrite('./gt_depth/' + str(batch_idx * 2 + 0) + ".png", gt_depth1)
                # cv2.imwrite('./gt_depth/' + str(batch_idx * 2 + 1) + ".png", gt_depth2)
                #mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth)
        mean_losses=total_loss/len(pbar)
        with open(self.output_file, 'a') as f:
            # 将损失记录到文件中,一定要是a，不然会覆盖
            f.write(f"{mean_losses}\n")
                # 打印损失
        self.evaluator.print()

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            # writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.step)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data/outputs["pred_depth"][j].data.max(), self.step)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    # def load_model(self):
    #     model_dict = self.model.state_dict()
    #     pretrained_dict = torch.load(self.settings.load_weights_dir)
    #     # for key in pretrained_dict:
    #     #     if key in model_dict:
    #     #         continue
    #     #     else:
    #     #         break
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.model.load_state_dict(model_dict)



    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

