"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Unet_CSC.py
    @Time: 2025/5/24 10:45
    @Email: None
"""

import torch.nn as nn
import torch


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


class UNet_CSC_V3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(UNet_CSC_V3, self).__init__()
        self.model_name = 'UNet_CSC_V3'

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
        # self.bottleneck = ConvBlock(base_c * 8, base_c * 16)
        self.csc_block = CSC_block(base_c * 8)

        self.csc_conv = nn.Conv2d(base_c * 8, base_c * 16, kernel_size=3, padding=1)

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
        x2 = self.enc2(self.pool1(x1))  # [1, 128, 128, 128]
        x3 = self.enc3(self.pool2(x2))  # [1, 256, 64, 64]
        x4 = self.enc4(self.pool3(x3))  # [1, 512, 32, 32]
        # # csc block 替换 Bottleneck

        x5 = self.csc_block(self.pool4(x4))
        x5 = self.csc_conv(x5)

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
        ker = 15
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

        # ow_13 = self.dw_13(res)
        # ow_31 = self.dw_31(res)
        # ow_33 = self.dw_33(res)
        # ow_11 = self.dw_11(res)
        # print(ow_13.shape, ow_31.shape, ow_33.shape, ow_11.shape)

        out = (
                self.dw_13(res) +
                self.dw_31(res) +
                self.dw_33(res) +
                self.dw_11(res)
        )
        out = res + out
        out = self.act(out)
        return self.out_conv(out)

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
