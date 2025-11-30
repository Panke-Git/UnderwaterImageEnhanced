"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Unet_HybridAttention.py
    @Time: 2025/6/2 01:29
    @Email: None
"""

import torch
import torch.nn as nn
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


class UNetHybridAttentionV26(nn.Module):
    """
    基于UNetHybridAttentionV25，优化高频去噪模块；
    """

    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(UNetHybridAttentionV26, self).__init__()
        self.model_name = 'UNetHybridAttentionV26'

        # Down path
        # Layer1
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        # Layer2
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Layer3
        # self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.hybrid_attention1 = HybridAttention(base_c * 2)
        self.conv1 = nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_c * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Layer4
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)

        # Up path
        # Layer4
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)
        # Layer3
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.hybrid_attention2 = HybridAttention(base_c * 4)
        self.conv2 = nn.Conv2d(base_c * 8, base_c * 4, kernel_size=3, padding=1)
        # self.dec3 = ConvBlock(base_c * 8, base_c * 4)
        # Layer2
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)
        # Layer1
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        # Output
        self.out_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.hybrid_attention1(self.pool2(x2))
        x3 = self.bn1(self.conv1(x3))
        # x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.hybrid_attention2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        # x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        out = self.out_conv(x)
        out = torch.sigmoid(out)  # Normalize output to [0,1]
        return out


class HybridAttention(nn.Module):
    def __init__(self, dim, threshold=0.05):
        super(HybridAttention, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        # self.threshold = threshold if isinstance(threshold, float) else nn.Parameter(torch.tensor(threshold))
        self.hf_denoiser=MultiDirHF_Denoiser(dim)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))

        self.norm1 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.norm2 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dw_ffn = DW_FFN(dim)

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def soft_threshold(self, x, thresh):
        return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)

    def forward(self, x):
        # 上边的分支，DWT部分
        Yl, Yh = self.dwt(x)
        ll = Yl * self.alpha
        Yh = self.hf_denoiser(Yh)
        # for j in range(len(Yh)):
        #     Yh[j] = self.soft_threshold(Yh[j], self.threshold)
        D_x = torch.abs(self.idwt((ll, Yh)))

        # 下边的BN+DW_FFN部分
        N_x = self.avg_pool(self.norm1(x))
        N_x = x + N_x
        N_x = self.dw_ffn(self.norm2(N_x))

        # 门控融合
        concat_x = torch.cat([D_x, N_x], dim=1)
        G = self.gate(concat_x)
        out = G * D_x + (1 - G) * N_x

        return out


class DW_FFN(nn.Module):
    def __init__(self, in_dim, expansion=2):
        super(DW_FFN, self).__init__()
        hidden_dim = in_dim * expansion

        self.expand = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)
        self.depthwise = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, stride=1,
                                   groups=hidden_dim)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden_dim, in_dim, kernel_size=1)

    def forward(self, x):
        return self.project(self.act(self.depthwise(self.expand(x))))


class GNConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super(GNConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HF_DenoiseBlock(nn.Module):
    def __init__(self, channel):
        super(HF_DenoiseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=2, dilation=2)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=channel)
        self.relu = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//8, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = self.relu(self.gn(out1 + out2))
        se = self.se(out)
        out = out * se

        return x - out


class MultiDirHF_Denoiser(nn.Module):
    def __init__(self, channel):
        super(MultiDirHF_Denoiser, self).__init__()
        self.lh_block = HF_DenoiseBlock(channel)
        self.hl_block = HF_DenoiseBlock(channel)
        self.hh_block = HF_DenoiseBlock(channel)

    def forward(self, x):
        Yh0 = x[0]

        lh = Yh0[:, :, 0, :, :]
        hl = Yh0[:, :, 1, :, :]
        hh = Yh0[:, :, 2, :, :]

        lh = self.lh_block(lh)
        hl = self.hl_block(hl)
        hh = self.hh_block(hh)

        Yh_denoised = torch.stack([lh, hl, hh], dim=2)
        return [Yh_denoised]
