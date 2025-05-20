"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: train_utils.py
    @Time: 2025/5/20 22:37
    @Email: None
"""
import os
import random
import torch
import numpy as np


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




