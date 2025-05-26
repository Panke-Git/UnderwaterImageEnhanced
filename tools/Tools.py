"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: Tools.py
    @Time: 2025/5/23 23:30
    @Email: None
"""
from src.utils.train_utils import generate_experiment_id

expt_id = generate_experiment_id(model='Unet',
                                 dataset='LSUI',
                                 loss='L1SSIM',
                                 note='SmoothL1Loss')
print(expt_id)
