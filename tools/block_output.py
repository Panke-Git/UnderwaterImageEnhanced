"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: block_output.py
    @Time: 2025/5/24 11:09
    @Email: None
"""


import torch
from torch import nn

from src import models
from src.models import LargeKernel as lk

from src.models.Unet_HybridAttention import HybridAttention
from src.models.block.MPNCOV import MPNCOV


x = torch.randn(1, 3, 256, 256).to(torch.device('cuda:0'))

# block = CSC_block(3)
# unet = lk.UIR_PolyKernel().to(torch.device('cuda:0'))
# unet = MPNCOV(input_dim=256, iterNum=5, dimension_reduction=256).to(torch.device('cuda:0'))
net = lk.UNetLKC().to(torch.device('cuda:0'))


print(x.shape)
out = net(x)
print(out.shape)


