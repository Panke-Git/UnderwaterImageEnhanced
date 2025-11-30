"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Unet_CSC.py
    @Time: 2025/5/24 10:45
    @Email: None
"""
import cv2
import numpy as np
import torch.nn as nn
import torch
# import pywt
from pyexpat import features

from pytorch_wavelets import DWTForward, DWTInverse


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


class UNet_DWT_V1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        """
        将离散小波变换添加到第一个跳跃连接中，然后经过DWT变换，后使用CLACHE处理低频通道，使用增强处理高频通道后输出
        :param in_channels:
        :param out_channels:
        :param base_c:
        """
        super(UNet_DWT_V1, self).__init__()
        self.model_name = 'UNet_DWT_V1'

        # Down path
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)

        self.dwt_csc_block = DWT_channel_block()

        # self.csc_block = CSC_block(base_c)

        # Up path
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        # Output
        self.out_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # [1, 64, 256, 256]
        x1 = self.dwt_csc_block(x1)
        x2 = self.enc2(self.pool1(x1))  # [1, 128, 128, 128]
        x3 = self.enc3(self.pool2(x2))  # [1, 256, 64, 64]
        x4 = self.enc4(self.pool3(x3))  # [1, 512, 32, 32]
        # # csc block 替换 Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        out = self.out_conv(x)
        out = torch.sigmoid(out)  # Normalize output to [0,1]
        return out


class CSC_block(nn.Module):
    def __init__(self, dim):
        super(CSC_block, self).__init__()
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
        res = self.in_conv(x)

        out = (
                self.dw_13(res) +
                self.dw_31(res) +
                self.dw_33(res) +
                self.dw_11(res)
        )
        out = res + out
        out = self.act(out)
        return self.out_conv(out)


class DWT_channel_block(nn.Module):
    def __init__(self, wave='db1', high_freq_gain=1.5):
        super(DWT_channel_block, self).__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.idwt = DWTInverse(wave=wave, mode='zero')
        self.gain = high_freq_gain

    def _process_LL_with_clahe(self, LL):
        # LL: [B, C, H, W] tensor
        B, C, H, W = LL.shape
        LL_np = LL.detach().cpu().numpy()
        LL_enhanced = np.zeros_like(LL_np)

        for b in range(B):
            for c in range(C):
                ll_img = LL_np[b, c]
                ll_norm = cv2.normalize(ll_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ll_clahe = self.clahe.apply(ll_norm)
                ll_clahe = ll_clahe.astype(np.float32) / 255.0
                ll_clahe *= np.max(ll_img)  # 保留动态范围
                LL_enhanced[b, c] = ll_clahe

        return torch.from_numpy(LL_enhanced).to(LL.device).type_as(LL)

    def forward(self, x):
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            raise ValueError("输入尺寸必须为偶数")
        Yl, Yh = self.dwt(x)
        LL = self._process_LL_with_clahe(Yl)

        Yh_enhanced = []
        for hf in Yh:
            hf = hf * self.gain  # [B, C, 3, H/2, W/2]
            Yh_enhanced.append(hf)

        out = self.idwt((LL, Yh_enhanced))
        return out

# class DWT_channel_block(nn.Module):
#     def __init__(self, wavelet='db1', high_freq_grain=1.5):
#         super(DWT_channel_block, self).__init__()
#         self.wavelet = wavelet
#         self.high_freq_grain = high_freq_grain
#         self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#
#     def _process_single_channel(self, channel_np):
#         coeffs2 = pywt.dwt2(channel_np, self.wavelet)
#         LL, (LH, HL, HH) = coeffs2
#
#         LL_norm = cv2.normalize(LL, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#         LL_clahe = self.clahe.apply(LL_norm)
#         LL_enhanced = LL_clahe.astype(np.float32) / 255.0
#         LL_enhanced *= np.max(LL)
#
#         LH *= self.high_freq_grain
#         HL *= self.high_freq_grain
#         HH *= self.high_freq_grain
#
#         enhanced = pywt.idwt2((LL_enhanced, (LH, HL, HH)), self.wavelet)
#         enhanced = np.clip(enhanced, 0, 255)
#         return enhanced.astype(np.float32)
#
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_np = x.detach().cpu().numpy()
#
#         output = np.zeros_like(x_np)
#         for b in range(B):
#             for c in range(C):
#                 feature = x_np[b, c]
#                 if H%2 != 0 or W%2 != 0:
#                     raise ValueError("H is not a multiple of W")
#                 output[b, c] = self._process_single_channel(feature)
#
#         return torch.from_numpy(output).to(x.device).type_as(x)
#
#

#
# class CSC_Block(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#
#         ker = 31
#         pad = ker // 2
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
#             nn.GELU()
#         )
#         self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
#         # Horizontal Strip Convolution
#         self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
#         # Vertical Strip Convolution
#         self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
#         # Square Kernel Convolution
#         self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
#         self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
#
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         out = self.in_conv(x)
#
#         out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
#         out = self.act(out)
#         return self.out_conv(out)
