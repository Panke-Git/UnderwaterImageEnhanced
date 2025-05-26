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

x = torch.randn(1, 3, 256, 256)

# block = CSC_block(3)
unet = models.UNet_CSC_V3(in_channels=3, out_channels=3, base_c=64)


out = unet(x)


