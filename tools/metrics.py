"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: metrics.py
    @Time: 2025/8/17 17:19
    @Email: None
"""


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 指标：torchmetrics & skimage
from torchmetrics.functional.image import (
    structural_similarity_index_measure as tm_ssim,
    peak_signal_noise_ratio as tm_psnr,
)
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

import lpips
from src.data.dataset import DataReader
from src import models as mode  # 按你的项目结构导入模型

# ----------------- 单张图像对齐评估 -----------------
def eval_one_image(pred_01_NCHW: torch.Tensor,
                   tgt_01_NCHW: torch.Tensor,
                   win: int = 11, sigma: float = 1.5, data_range: float = 1.0):
    """
    同一张图同时计算 torchmetrics & skimage 的 PSNR/SSIM。
    输入:
      pred_01_NCHW, tgt_01_NCHW: [1,3,H,W], 浮点, [0,1]
    返回:
      psnr_tm, ssim_tm, psnr_sk, ssim_sk   (float)
    """
    # 统一范围
    pred = pred_01_NCHW.clamp(0, 1)
    tgt  = tgt_01_NCHW.clamp(0, 1)

    # --- torchmetrics（逐图像）---
    psnr_tm = tm_psnr(pred, tgt, data_range=data_range).item()
    ssim_tm = tm_ssim(pred, tgt, data_range=data_range, kernel_size=win, sigma=sigma).item()

    # --- skimage（逐图像）---
    pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # HWC
    tgt_np  = tgt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    H, W, _ = tgt_np.shape
    win_size = min(win, H, W)
    if win_size % 2 == 0:
        win_size -= 1

    psnr_sk = sk_psnr(tgt_np, pred_np, data_range=data_range)
    ssim_sk = sk_ssim(
        tgt_np, pred_np,
        data_range=data_range,
        channel_axis=-1,
        win_size=win_size,
        gaussian_weights=True,
        sigma=sigma
    )

    return psnr_tm, ssim_tm, psnr_sk, ssim_sk


def main():
    # ----------------- 初始化模型 -----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = mode.UNet().to(device)
    state_dict = torch.load(
        r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNet\20250815_190500\best_result\TOP_PSNR.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # ----------------- 验证集 -----------------
    val_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val'
    val_dataset = DataReader(
        img_dir=val_dir,
        input='input',
        target='target',
        mode='test',
        ori=False,
        img_options={'w': 256, 'h': 256}
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # ----------------- LPIPS -----------------
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    def normalize_for_lpips(x):  # [0,1] -> [-1,1]
        return x * 2 - 1

    # ----------------- 累计器 -----------------
    total = {
        'psnr_tm': 0.0, 'ssim_tm': 0.0,
        'psnr_sk': 0.0, 'ssim_sk': 0.0,
        'lpips':   0.0,
    }
    nimg = 0

    # ----------------- 测试循环（逐图像→按总图像数平均） -----------------
    with torch.inference_mode():
        for data in tqdm(val_loader, desc="Evaluating"):
            input_img, target_img = data[0], data[1]  # [B, 3, H, W]
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output = model(input_img)
            B = output.size(0)
            for i in range(B):
                pred = output[i:i+1]   # [1,3,H,W]
                gt   = target_img[i:i+1]   # [1,3,H,W]

                # —— PSNR/SSIM：tm & skimage（对齐参数）——
                psnr_tm, ssim_tm, psnr_sk, ssim_sk = eval_one_image(pred, gt, win=11, sigma=1.5, data_range=1.0)
                total['psnr_tm'] += psnr_tm
                total['ssim_tm'] += ssim_tm
                total['psnr_sk'] += psnr_sk
                total['ssim_sk'] += ssim_sk

                # —— LPIPS ——（需要 [-1,1]）
                lp = lpips_fn(normalize_for_lpips(pred), normalize_for_lpips(gt)).item()
                total['lpips'] += lp

            nimg += B

    # ----------------- 输出结果（按图像数平均） -----------------
    for k in total:
        total[k] /= max(1, nimg)

    print('--------------------------------')
    print(f"TM  Average PSNR : {total['psnr_tm']:.4f} dB")
    print(f"TM  Average SSIM : {total['ssim_tm']:.4f}")
    print(f"SK  Average PSNR : {total['psnr_sk']:.4f} dB")
    print(f"SK  Average SSIM : {total['ssim_sk']:.4f}")
    print(f"Avg LPIPS        : {total['lpips']:.4f}")


if __name__ == '__main__':
    main()


