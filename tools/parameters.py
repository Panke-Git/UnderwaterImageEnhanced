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

model = models.UNetHybridAttentionV23().to('cuda:0')
summary(model, input_size=(1, 3, 256, 256), depth=3)



from ptflops import get_model_complexity_info
import torch

model = models.UNetHybridAttentionV23().to('cuda:0')
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(model, (3, 256, 256),
                                              as_strings=True,
                                              print_per_layer_stat=False)
print(f'FLOPs: {flops}')
print(f'Params: {params}')
