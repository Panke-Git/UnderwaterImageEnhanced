"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: tm_sk_PSNR.py
    @Time: 2025/8/17 17:07
    @Email: None
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
from torchmetrics.functional.image import peak_signal_noise_ratio as tm_psnr

def to_numpy_img(x):  # NCHW [0,1] float -> HWC float32
    return x.squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.float32)

def eval_one_image(pred_01_NCHW, tgt_01_NCHW, win=11, sigma=1.5, data_range=1.0):
    # clamp + 同一范围
    pred = pred_01_NCHW.clamp(0,1)
    tgt  = tgt_01_NCHW.clamp(0,1)

    # ---- torchmetrics (逐图像) ----
    # PSNR：返回标量
    psnr_tm = tm_psnr(pred, tgt, data_range=data_range).item()

    # SSIM：为对齐论文/ skimage，显式指定核与 sigma，并逐图像统计
    ssim_tm = tm_ssim(pred, tgt, data_range=data_range, kernel_size=win, sigma=sigma).item()

    # ---- skimage (逐图像) ----
    pred_np = to_numpy_img(pred)
    tgt_np  = to_numpy_img(tgt)

    # win_size 不得超过短边，且要奇数
    H,W,_ = tgt_np.shape
    win_s = min(win, H, W)
    if win_s % 2 == 0: win_s -= 1

    psnr_sk = sk_psnr(tgt_np, pred_np, data_range=data_range)
    ssim_sk = sk_ssim(tgt_np, pred_np, data_range=data_range, channel_axis=-1,
                      win_size=win_s, gaussian_weights=True, sigma=sigma)

    return psnr_tm, ssim_tm, psnr_sk, ssim_sk


# 随便造一张预测和目标图 (这里是随机噪声，实际用模型输出和GT)
pred = torch.rand(1, 3, 256, 256)   # 模型输出
target = torch.rand(1, 3, 256, 256) # Ground Truth

psnr_tm, ssim_tm, psnr_sk, ssim_sk = eval_one_image(pred, target)

print("Torchmetrics PSNR:", psnr_tm)
print("Torchmetrics SSIM:", ssim_tm)
print("Skimage PSNR:", psnr_sk)
print("Skimage SSIM:", ssim_sk)