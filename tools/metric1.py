"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: metric1.py
    @Time: 2025/9/2 23:43
    @Email: None
"""
# coding=utf-8
"""
    @Project:
    @Author: Panke
    @FileName： metric.py  (aligned with training metrics)
    @Date：2025/7/23 15:05
    @Email: None
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)
from tqdm import tqdm
import numpy as np

from src.data.dataset import DataReader

# ✅ 替换为你的模型类路径
from src import models as mode  # 与训练保持一致

def main():
    # ----------------- 初始化模型 -----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = mode.UNetHybridAttentionV23_2().to(device)
    model.load_state_dict(torch.load(
        r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2\20250816_125830\best_result\TOP_PSNR.pth",
        map_location=device
    ))
    model.eval()

    # ----------------- 数据集 -----------------
    val_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\UIEB19\Val'
    val_dataset = DataReader(
        img_dir=val_dir,
        input='input',
        target='GT',
        mode='test',
        ori=False,
        img_options={'w': 256, 'h': 256}  # 与训练一致
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # ----------------- 评估（与训练完全一致口径） -----------------
    psnr_total = 0.0
    ssim_total = 0.0
    num_images = 0

    with torch.inference_mode():
        for data in tqdm(val_loader):
            inp, target = data[0].to(device), data[1].to(device)

            # 前向推理并限制到[0,1]，与训练一致
            pred = model(inp).clamp(0, 1)

            # PSNR：逐图计算再求平均（与训练相同写法）
            # 返回形状 [B] 的每图 PSNR
            psnr_each = peak_signal_noise_ratio(
                pred, target, data_range=1.0, dim=(1, 2, 3), reduction='none'
            )  # shape [B]
            psnr_total += psnr_each.sum().item()

            # SSIM：torchmetrics 返回的是 batch 平均（标量），乘以 B 再累加
            ssim_batch = structural_similarity_index_measure(
                pred, target, data_range=1.0
            )  # 标量（batch mean）
            B = pred.size(0)
            ssim_total += ssim_batch.item() * B

            num_images += B

    avg_psnr = psnr_total / max(1, num_images)
    avg_ssim = ssim_total / max(1, num_images)

    print('--------------------------------')
    print(f"✅ Average PSNR : {avg_psnr:.4f} dB")
    print(f"✅ Average SSIM : {avg_ssim:.4f}")

if __name__ == '__main__':
    main()
