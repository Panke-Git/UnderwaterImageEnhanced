# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： metric.py
    @Date：2025/7/23 15:05
    @Email: None
"""

import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.dataset import DataReader
import numpy as np

# ✅ 替换为你的模型类
from src import models as mode  # 请根据你实际模型路径修改

def main():
    # ----------------- 初始化模型 -----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = mode.UNetHybridAttentionV23_2_NoThreshold().to(device)
    model.load_state_dict(torch.load(
        r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2_NoThreshold\20250817_222332\best_result\TOP_PSNR.pth",
        map_location=device))
    model.eval()

    val_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val'

    val_dataset = DataReader(img_dir=val_dir,
                             input='input',
                             target='GT',
                             mode='test',
                             ori=False,
                             img_options={'w': 256, 'h': 256})

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True, )

    # ----------------- 初始化 LPIPS -----------------
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # ----------------- 工具函数 -----------------
    def normalize_for_lpips(x):
        return x * 2 - 1  # 从 [0,1] 转为 [-1,1]

    # ----------------- 评估指标累加器 -----------------
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    # ----------------- 测试循环 -----------------
    with torch.no_grad():
        for data in tqdm(val_loader):
            input_img, target_img = data[0], data[1]  # [B, 3, H, W]
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # 前向推理
            output = model(input_img)
            for i in range(output.size(0)):  # 支持 batch > 1
                pred = output[i].clamp(0, 1).unsqueeze(0)  # [1, 3, H, W]
                target = target_img[i].unsqueeze(0)

                # ---------- LPIPS ----------
                pred_lpips = normalize_for_lpips(pred)
                target_lpips = normalize_for_lpips(target)
                lp = lpips_fn(pred_lpips, target_lpips).item()
                total_lpips += lp

                # ---------- PSNR & SSIM ----------
                pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # HWC
                target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

                psnr = compare_psnr(target_np, pred_np, data_range=1.0)

                # 适配 win_size，避免尺寸过小时报错
                H, W, _ = target_np.shape
                win_size = min(7, H, W)
                if win_size % 2 == 0:
                    win_size -= 1

                ssim = compare_ssim(
                    target_np,
                    pred_np,
                    data_range=1.0,
                    channel_axis=-1,
                    win_size=win_size,
                    gaussian_weights=True,  # 推荐与经典实现对齐
                    sigma=1.5
                )

                total_psnr += psnr
                total_ssim += ssim
                count += 1

    # ----------------- 输出结果 -----------------
    # print("\nDataset: LSUI19, Model: UNet")
    print('--------------------------------')
    print(f"✅ Average PSNR : {total_psnr / count:.4f} dB")
    print(f"✅ Average SSIM : {total_ssim / count:.4f}")
    print(f"✅ Average LPIPS: {total_lpips / count:.4f}")

if __name__ == '__main__':
    main()

