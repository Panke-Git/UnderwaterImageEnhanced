"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Unet_HybridAttention.py
    @Time: 2025/6/2 01:29
    @Email: None
"""
import os

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F


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


class UnetHybridAttentionV23_2_Learnable(nn.Module):
    """
    基于UNetHybridAttentionV8，使用可学习阈值，输出每个高频子带的软阈值的稀疏性
    """

    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(UnetHybridAttentionV23_2_Learnable, self).__init__()
        self.model_name = 'UnetHybridAttentionV23_2_Learnable'

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
    def __init__(self, dim, threshold=0.05, enable_sparse_reg=False):
        """
        :param dim: 输入通道数
        :param threshold: 初始软阈值，可为 float 或初始化值
        :param enable_sparse_reg: 是否启用稀疏正则项（用于训练时调用）
        """
        super(HybridAttention, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')

        # ⬇️ 可学习阈值：用 ReLU + ε 保证最小阈值 > 0
        self.raw_threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.min_eps = 1e-2  # 阈值下限偏移量

        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))

        self.norm1 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.norm2 = GNConvBlock(in_ch=dim, out_ch=dim, num_groups=8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dw_ffn = DW_FFN(dim)

        # ⬇️ 缓存每个 Yh 的稀疏性，供 epoch 汇总
        self.sparsity_accumulator = []
        # self.log_path = log_path

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.enable_sparse_reg = enable_sparse_reg  # 是否启用稀疏损失

        # ⬇️ 存储稀疏性用于正则项（可被主 loss 函数引用）
        self.latest_sparse_loss = 0.0

    def get_effective_threshold(self):
        # ⬇️ 限制阈值始终 ≥ min_eps
        return F.relu(self.raw_threshold) + self.min_eps

    def soft_threshold(self, x, thresh):
        # ⬇️ 标准软阈值函数
        return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)

    def calc_sparsity(self, tensor, eps=1e-6):
        # ⬇️ 统计输出中“接近零”的比例
        total = tensor.numel()
        zero_count = (torch.abs(tensor) < eps).sum().item()
        return zero_count / total

    def forward(self, x):
        threshold = self.get_effective_threshold()  # ⬅️ 动态阈值获取

        # ================= 上支 DWT 分支 =================
        Yl, Yh = self.dwt(x)
        ll = Yl * self.alpha
        batch_sparsity = []

        for j in range(len(Yh)):
            Yh[j] = self.soft_threshold(Yh[j], threshold)

            if self.training:
                sparsity = self.calc_sparsity(Yh[j])
                batch_sparsity.append(sparsity)
        if self.training and batch_sparsity:
            self.sparsity_accumulator.append(batch_sparsity)

        D_x = torch.abs(self.idwt((ll, Yh)))

        # ================ 下支 FFN 分支 ===================
        N_x = self.avg_pool(self.norm1(x))
        N_x = x + N_x
        N_x = self.dw_ffn(self.norm2(N_x))

        # ================ 门控融合 ========================
        concat_x = torch.cat([D_x, N_x], dim=1)
        G = self.gate(concat_x)
        out = G * D_x + (1 - G) * N_x

        # ================ 可选稀疏正则项 ===================
        # if self.enable_sparse_reg and len(sparse_losses) > 0:
        #     self.latest_sparse_loss = sum(sparse_losses) / len(sparse_losses)
        # else:
        #     self.latest_sparse_loss = 0.0

        return out

    def log_epoch_stats(self, epoch_num, log_path):
        """每个 epoch 结束时调用此函数，将平均稀疏度 + threshold 写入文件"""
        if not self.sparsity_accumulator:
            return

        # 转置累积列表：batch × 3 → 3 × batch
        Yh_by_subband = list(zip(*self.sparsity_accumulator))  # Yh[0~2]
        avg_sparsities = [sum(x) / len(x) for x in Yh_by_subband]
        threshold_val = self.get_effective_threshold().item()

        # 写入日志
        with open(log_path, "a") as f:
            f.write(f"[Epoch {epoch_num}] Threshold: {threshold_val:.4f}\n")
            for j, s in enumerate(avg_sparsities):
                f.write(f"    Yh[{j}] avg sparsity: {s:.4f}\n")
            f.write("\n")

        # 清空缓存
        self.sparsity_accumulator.clear()


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
