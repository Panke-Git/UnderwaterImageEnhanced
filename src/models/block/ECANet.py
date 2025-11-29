"""
    @Project: UnderwaterImageEnhanced
    @Author: paxton
    @FileName： ECANet.py
    @Date：2025/11/29 09:19
    @OS：
    @Email: None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA)
    输入:  x: (B, C, H, W)
    输出:  y: (B, C, H, W)，每个通道有一个注意力权重
    """
    def __init__(self, channels, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)        # (B,C,H,W)->(B,C,1,1)
        # 1D conv 做通道间局部交互：输入通道=1，输出通道=1，kernel=k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # 全局平均池化 → (B,C,1,1)
        y = self.avg_pool(x)                        # (B,C,1,1)
        y = y.view(B, 1, C)                         # (B,1,C)

        # 1D conv 进行通道间信息交互
        y = self.conv(y)                            # (B,1,C)
        y = self.sigmoid(y)                         # (B,1,C)

        y = y.view(B, C, 1, 1)                      # (B,C,1,1)
        return x * y                                # 通道加权后输出
