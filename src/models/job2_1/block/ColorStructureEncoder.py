"""
    @Project: UnderwaterImageEnhanced
    @Author: paxton
    @FileName： ColorStructureEncoder.py
    @Date：2025/11/29 00:30
    @OS：
    @Email: None
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F



def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""

    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)




class ColorStructureEncoder(nn.Module):
    """
    输入:  x: (B,3,H_in,W_in) 例如 (B,3,256,256)
    内部:
        - WavePool -> LL (B,3,H_ll,W_ll)
        - 在 LL 上做 RGB / LAB / HSV
        - 通道拼接成 (B,9,H_ll,W_ll)
        - 再插值回 (B,9,H_in,W_in)

    输出:
        structure: (B,9,H_in,W_in)
    """

    def __init__(self, use_wavepool=True):
        super(ColorStructureEncoder, self).__init__()
        self.use_wavepool = use_wavepool
        if use_wavepool:
            self.wave = WavePool(3)  # 你原来的小波池化类

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B,3,H_in,W_in)
        返回:
            structure: (B,9,H_in,W_in)
        """
        B, C, H_in, W_in = x.shape

        # 1) WavePool 取 LL 低频
        if self.use_wavepool:
            LL, LH, HL, HH = self.wave(x)   # LL: (B,3,H_ll,W_ll)
        else:
            LL = x                          # 不用波小波的话，直接用原图

        B_ll, C_ll, H_ll, W_ll = LL.shape

        # 2) 转为 numpy，做 RGB/LAB/HSV
        LL_np = LL.permute(0, 2, 3, 1).cpu().numpy()  # (B,H_ll,W_ll,3)
        structs = []

        for b in range(B_ll):
            ll = LL_np[b].astype(np.float32)          # (H_ll,W_ll,3)

            ll_rgb = ll
            ll_lab = cv2.cvtColor(ll, cv2.COLOR_RGB2LAB)
            ll_hsv = cv2.cvtColor(ll, cv2.COLOR_RGB2HSV)

            # 通道拼接: (H_ll,W_ll,9)
            stru = np.concatenate([ll_rgb, ll_lab, ll_hsv], axis=-1)

            # 转为 tensor: (9,H_ll,W_ll)
            stru = torch.from_numpy(stru).permute(2, 0, 1)  # C,H,W
            structs.append(stru)

        # 3) 堆叠 batch 维 -> (B,9,H_ll,W_ll)
        structure = torch.stack(structs, dim=0).to(x.device).float()

        # 4) 插值回输入分辨率 (B,9,H_in,W_in)
        if H_ll != H_in or W_ll != W_in:
            structure = F.interpolate(
                structure,
                size=(H_in, W_in),
                mode="bilinear",
                align_corners=False
            )

        return structure