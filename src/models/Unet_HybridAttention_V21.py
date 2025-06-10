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


class UNetHybridAttentionV20(nn.Module):
    """
    基于UNetHybridAttentionV8，将双流拆分，保留上半部分的DWT的部分；其中没有残差；
    """

    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(UNetHybridAttentionV20, self).__init__()
        self.model_name = 'UNetHybridAttentionV20'

        # Down path
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        # self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.hybrid_attention1 = HybridAttention(base_c * 2)
        self.conv1 = nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)

        # Up path
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.hybrid_attention2 = HybridAttention(base_c * 4)
        self.conv2 = nn.Conv2d(base_c * 8, base_c * 4, kernel_size=3, padding=1)
        # self.dec3 = ConvBlock(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)

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
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.threshold = threshold if isinstance(threshold, float) else nn.Parameter(torch.tensor(threshold))
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))

        self.norm1 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.norm2 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dw_ffn = DW_FFN(dim)

    def soft_threshold(self, x, thresh):
        return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)

    def forward(self, x):
        Yl, Yh = self.dwt(x)
        ll = Yl * self.alpha

        for j in range(len(Yh)):
            Yh[j] = self.soft_threshold(Yh[j], self.threshold)

        x1 = self.avg_pool(self.norm1(x))
        x2 = x + x1
        x3 = self.dw_ffn(self.norm2(x2))

        out = x3 + torch.abs(self.idwt((ll, Yh)))

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


def to_3d(x):
    B, C, H, W = x.shape
    return x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]


def to_4d(x, H, W):
    B, N, C = x.shape
    return x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, mode='biasfree'):
        super(LayerNorm, self).__init__()
        if mode == 'biasfree':
            self.norm = BiasFree_LayerNorm(dim)
        elif mode == 'bias':
            self.norm = WithBias_LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported LayerNorm mode: {mode}")

    def forward(self, x):
        H, W = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), H, W)
