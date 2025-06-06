"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Unet_CSC_LKA.py
    @Time: 2025/6/5 23:36
    @Email: None
"""
import numbers

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat * 2 * 2, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))

        return x



class CSC_Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        # Horizontal Strip Convolution
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        # Vertical Strip Convolution
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        # Square Kernel Convolution
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        out = self.act(out)
        return self.out_conv(out)


class ConvBlock(nn.Module):
    """Conv => BN => ReLU x2"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetCSC_LKA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=36):
        super(UNetCSC_LKA, self).__init__()
        self.model_name = 'UNetCSC_LKA'

        # Down path
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = Downsample(base_c)
        self.lka1 = LKABlock(base_c)
        # self.enc2 = ConvBlock(base_c, base_c * 2)
        # self.pool2 = nn.MaxPool2d(2)

        self.enc3 = Downsample(base_c * 2)
        self.lka2 = LKABlock(base_c * 2)
        # self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        # self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        # self.bottleneck = ConvBlock(base_c * 4, base_c * 4)
        self.bottleneck = CSC_Block(base_c * 4)

        self.reduce_chan_level3 = nn.Conv2d(int(base_c * 2 ** 3), int(base_c * 2 ** 2), kernel_size=1, bias=False)
        self.decoder_level3 = LKABlock(int(base_c * 2 ** 2))
        self.up3_2 = Upsample(int(base_c * 2 ** 2))

        # self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=3, stride=1, padding=1)
        # self.dec3 = ConvBlock(base_c * 8, base_c * 4)

        self.reduce_chan_level2 = nn.Conv2d(int(base_c * 2 ** 2), int(base_c * 2 ** 1), kernel_size=1, bias=False)
        self.decoder_level2 = LKABlock(int(base_c * 2 ** 1))
        self.up2_1 = Upsample(int(base_c * 2 ** 1))
        # self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        # self.dec2 = ConvBlock(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        # Output
        self.out_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # 36, 256, 256
        x2 = self.enc2(self.lka1(x1))  # 72, 128, 128
        x3 = self.enc3(self.lka2(x2))  # 144, 64, 64
        # Bottleneck
        x5 = self.bottleneck(x3)

        inp_dec_level3 = cat(x5, x3)  # [288, 64, 64]
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # [144, 64, 64]
        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # [144, 64, 64]

        inp_dec_level2 = self.up3_2(out_dec_level3)  # [72, 128, 128]
        inp_dec_level2 = cat(inp_dec_level2, x2)  # [144, 128, 128]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [72, 128, 128]
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [72, 128, 128]

        # x = self.up1(x)
        x = self.up2_1(out_dec_level2)
        x = self.dec1(torch.cat([x, x1], dim=1))

        out = self.out_conv(x)
        out = torch.sigmoid(out)  # Normalize output to [0,1]
        return out


def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x
