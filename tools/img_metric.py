"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: img_metric.py
    @Time: 2025/9/7 00:45
    @Email: None
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
import src.models as models

# ========== 工具函数 ==========
def ensure_rgb(bgr):
    if bgr is None:
        return None
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def img_to_tensor01(rgb):
    # HWC(0..255) -> 1xCxHxW(0..1)
    return transforms.ToTensor()(rgb).unsqueeze(0)

# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="Pure evaluation: per-image PSNR & SSIM to Excel")
    parser.add_argument("--input_dir", default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val\input', help="待增强输入图片文件夹")
    parser.add_argument("--gt_dir", default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val\GT', help="GT图片文件夹（与输入同名一一对应）")
    parser.add_argument("--pth", default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2\20250819_222858\best_result\TOP_PSNR.pth', help="模型权重 .pth 路径（state_dict）")
    parser.add_argument("--excel", default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val\result.xlsx', help="输出 Excel 路径（默认 metrics_时间.xlsx）")
    parser.add_argument("--size", type=int, default=256, help="评估时统一缩放尺寸，默认 256")
    args = parser.parse_args()

    infer_size = (args.size, args.size)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = args.excel or str(Path(args.input_dir).parent / f"metrics_{ts}.xlsx")
    print(excel_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # 1) 准备模型（与训练一致）
    ModelCtor = models.UNetHybridAttentionV23_2  # 如需换模型只改这里
    model = ModelCtor().to(device)
    state = torch.load(args.pth, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) 收集文件
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    inputs = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in exts])
    if not inputs:
        raise RuntimeError(f"在 {args.input_dir} 未找到图像。")

    rows = []
    ok, skip = 0, 0

    for ip in inputs:
        name = ip.name
        gt_path = Path(args.gt_dir) / name
        if not gt_path.exists():
            print(f"[SKIP] GT 缺失：{name}")
            skip += 1
            continue

        img_rgb = ensure_rgb(cv2.imread(str(ip)))
        gt_rgb  = ensure_rgb(cv2.imread(str(gt_path)))
        if img_rgb is None or gt_rgb is None:
            print(f"[SKIP] 读图失败：{name}")
            skip += 1
            continue

        # 统一缩放到 size×size（与训练/验证一致）
        img_256 = cv2.resize(img_rgb, infer_size, interpolation=cv2.INTER_AREA)
        gt_256  = cv2.resize(gt_rgb,  infer_size, interpolation=cv2.INTER_AREA)

        x  = img_to_tensor01(img_256).to(device)  # [1,C,H,W], 0..1
        gt = img_to_tensor01(gt_256).to(device)

        # 推理
        with torch.no_grad():
            pred = model(x).clamp(0, 1)

        # 指标（逐图）
        psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0, dim=(1, 2, 3), reduction="none").item()
        ssim = structural_similarity_index_measure(pred, gt, data_range=1.0).item()

        rows.append({"filename": name, "PSNR": float(psnr), "SSIM": float(ssim)})
        ok += 1
        print(f"[OK] {name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")

    # 3) 写Excel（三列）
    if not rows:
        raise RuntimeError("没有可写入的评估结果。")
    df = pd.DataFrame(rows, columns=["filename", "PSNR", "SSIM"])
    df.to_excel(excel_path, index=False)
    print(f"\n✅ 评估完成：成功 {ok}，跳过 {skip}。结果已写入：{excel_path}")

if __name__ == "__main__":
    main()



