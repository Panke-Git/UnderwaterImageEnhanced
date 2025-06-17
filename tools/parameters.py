"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: parameters.py
    @Time: 2025/6/16 22:25
    @Email: None
"""

from torchinfo import summary
import torch
from src import models
from fvcore.nn import FlopCountAnalysis

model = models.UNetHybridAttentionV25().to('cuda:0')
summary(model, input_size=(1, 3, 256, 256), depth=3)



