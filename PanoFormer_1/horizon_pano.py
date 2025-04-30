"""
## PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation
## Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, Yao Zhao
## https://arxiv.org/abs/2203.09283
## The code is reproducted based on uformer:https://github.com/ZhendongWang6/Uformer
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from network.PSA import *

from network.equisamplingpoint import genSamplingPattern


class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels):
        super(StripPooling, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        # self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.ac = nn.Sigmoid()
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        out = self.conv3(x1 + x2)
        out_att = self.ac(out)
        return out_att


#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


#########################################
########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., flag=0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())
        # self.hw = StripPooling(hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = H

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)
        # bs,hidden_dim,32x32
        # att = self.hw(x)

        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)

        # x = x * att

        # x = self.active(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x


#########################################
# Downsample Block
#这里是一定要batchnorm的，不然downsample之后得到的方差实在是太大了
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.ReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=0),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')  # for 1024*512
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # val=torch.var(x,dim=[2,3])
        return x


#########################################
########### LeWinTransformer #############
class PanoformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False, ref_point=None, flag=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point  # generate_ref_points(self.input_resolution[1], self.input_resolution[0])
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)

        self.dattn = PanoSelfAttention(num_heads, dim, k=9, last_feat_height=self.input_resolution[0],
                                       last_feat_width=self.input_resolution[1], scales=1, dropout=0, need_attn=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, flag=flag)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        x = self.dattn(x, x.unsqueeze(0), self.ref_point.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


########### Basic layer of Uformer ################
class BasicPanoformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False, ref_point=None, flag=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            PanoformerBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, win_size=win_size,
                            shift_size=0 if (i % 2 == 0) else win_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                            se_layer=se_layer, ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


########### Uformer ################
class Panoformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=256, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[8, 16, 32, 64, 128, 128, 64, 32, 16],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        # self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()#torch.load("network6/Equioffset256x512.pth")
        self.ref_point128x256 = genSamplingPattern(128, 256, 3,
                                                   3).cuda()  # torch.load("network6/Equioffset128x256.pth")
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()  # torch.load("network6/Equioffset64x128.pth")
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()  ##torch.load("network6/Equioffset32x64.pth")
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()  # torch.load("network6/Equioffset16x32.pth")
        self.ref_point8x16 = genSamplingPattern(8, 16, 3, 3).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        # self.pre_block = PreprocBlock(in_channels=3, out_channels=64, kernel_size_lst=[[3, 9], [5, 11], [5, 7], [7, 7]])

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=2,
                                    act_layer=nn.GELU)  # stride = 2 for 1024*512
        # self.conv_in=nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.GELU()
        # )
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))
        self.output_proj_0 = OutputProj(in_channel=2 * embed_dim, out_channel=8 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // 2, img_size // 2 * 2))
        self.output_proj_1 = OutputProj(in_channel=4 * embed_dim, out_channel=16 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 * 2), img_size // (2 ** 2) * 2))
        self.output_proj_2 = OutputProj(in_channel=8 * embed_dim, out_channel=32 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 ** 3), img_size // (2 ** 3) * 2))
        self.output_proj_3 = OutputProj(in_channel=16 * embed_dim, out_channel=64 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 ** 4), img_size // (2 ** 4) * 2))
        # Encoder
        self.encoderlayer_0 = BasicPanoformerLayer(dim=embed_dim,
                                                   output_dim=embed_dim,
                                                   input_resolution=(img_size, img_size * 2),
                                                   depth=depths[0],
                                                   num_heads=num_heads[0],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point128x256, flag=0)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2, input_resolution=(img_size, img_size * 2))
        self.encoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                   output_dim=embed_dim * 2,
                                                   input_resolution=(img_size // 2, img_size * 2 // 2),
                                                   depth=depths[1],
                                                   num_heads=num_heads[1],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point64x128, flag=0)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.encoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                   output_dim=embed_dim * 4,
                                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                   depth=depths[2],
                                                   num_heads=num_heads[2],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point32x64, flag=0)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8,
                                     input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.encoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                   output_dim=embed_dim * 8,
                                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                   depth=depths[3],
                                                   num_heads=num_heads[3],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point16x32, flag=0)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16,
                                     input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))

        # Bottleneck
        self.conv = BasicPanoformerLayer(dim=embed_dim * 16,
                                         output_dim=embed_dim * 16,
                                         input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
                                         depth=depths[4],
                                         num_heads=num_heads[4],
                                         win_size=win_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=conv_dpr,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint,
                                         token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer,
                                         ref_point=self.ref_point8x16, flag=0)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = BasicPanoformerLayer(dim=embed_dim * 16,
                                                   output_dim=embed_dim * 16,
                                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                   depth=depths[5],
                                                   num_heads=num_heads[5],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[:depths[5]],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point16x32, flag=1)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                   output_dim=embed_dim * 8,
                                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                   depth=depths[6],
                                                   num_heads=num_heads[6],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point32x64, flag=1)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                   output_dim=embed_dim * 4,
                                                   input_resolution=(img_size // 2, img_size * 2 // 2),
                                                   depth=depths[7],
                                                   num_heads=num_heads[7],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point64x128, flag=1)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                   output_dim=embed_dim * 2,
                                                   input_resolution=(img_size, img_size * 2),
                                                   depth=depths[8],
                                                   num_heads=num_heads[8],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point128x256, flag=1)
        self.lstmdecoder=HorizonNet(use_rnn=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x):
        # Input Projection
        # y = self.pre_block(x)
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)
        #使用encoder的输出得到layout
        layout_feature = []
        layout_feature.append(self.output_proj_0(pool0))
        layout_feature.append(self.output_proj_1(pool1))
        layout_feature.append(self.output_proj_2(pool2))
        layout_feature.append(self.output_proj_3(pool3))
        bon,cor=self.lstmdecoder(layout_feature)
        # Bottleneck
        conv4 = self.conv(pool3)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection

        y = self.output_proj(deconv3)
        outputs = {}
        outputs["pred_depth"] = y
        return outputs

class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )

'''
HorizonNet
'''
class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self,  use_rnn):
        super(HorizonNet, self).__init__()
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512

        # Encoder
        # Inference channels number from each block of the encoder
        c1, c2, c3, c4 = [256,512,1024,2048]
        c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale
        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(0.5)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=3 * self.step_cols)
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        else:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)
    #这里要在panoformer的input的时候用
    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        # for i in conv_list:
        #     continue
        feature = self.reduce_height_module(x, 1024//self.step_cols)
        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor

