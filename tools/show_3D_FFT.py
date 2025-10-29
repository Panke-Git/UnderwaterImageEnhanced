"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_3D_FFT.py
    @Time: 2025/7/28 22:30
    @Email: None
"""
# infer_fft_2d3d_lfd.py
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 激活 3D
import matplotlib.pyplot as plt
import src.models as models

import torchvision.transforms as transforms


def fft2_gray(img_uint8):
    """img_uint8: HxW[x3] uint8 -> 返回 (F_shift复数矩阵, log|F| 用于绘图)"""
    if img_uint8.ndim == 3:
        gray = np.dot(img_uint8[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = img_uint8.astype(np.float32)
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0

    F_uv = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F_uv)
    mag = np.abs(F_shift)
    log_mag = np.log1p(mag)  # 避免 log(0)
    return F_shift, log_mag


def save_fft_2d(log_mag, path):
    log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-12)
    plt.figure(figsize=(6, 6))
    plt.imshow(log_mag_norm, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()


def save_fft_3d(log_mag, path, stride=1):
    """
    3D 曲面频谱图；stride>1 时对网格下采样，加快绘制
    """
    H, W = log_mag.shape
    yy = np.arange(0, H, stride)
    xx = np.arange(0, W, stride)
    X, Y = np.meshgrid(xx, yy)
    Z = log_mag[yy][:, xx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_title("3D FFT Spectrum (log magnitude)")
    ax.set_xlabel("Frequency X")
    ax.set_ylabel("Frequency Y")
    ax.set_zlabel("Log |F(u,v)|")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def compute_lfd(F_shift, radius_frac=0.10):
    """
    LFD = 低频能量 / 总能量；低频半径 = 最大可用半径 * radius_frac
    """
    H, W = F_shift.shape
    cy, cx = H // 2, W // 2
    power = np.abs(F_shift) ** 2
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(cy, cx)
    r = radius_frac * max_r
    low_energy = power[dist <= r].sum()
    total_energy = power.sum() + 1e-12
    return float(low_energy / total_energy), float(r), int(max_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2\20250819_222858\best_result\TOP_PSNR.pth', help="模型权重 .pth 路径（state_dict 或整模型）")
    parser.add_argument("--image", type=str, default=r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\tools\img\input.jpg",
                        help="输入图片路径（默认使用你给的示例路径）")
    parser.add_argument("--outdir", type=str, default="./out_fft", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="例如 cuda:0 / cpu")
    parser.add_argument("--weights-only", action="store_true", help="ckpt 是 state_dict 时加上")
    parser.add_argument("--radius-frac", type=float, default=0.10, help="LFD 低频半径占比，默认 0.10")
    parser.add_argument("--stride3d", type=int, default=2, help="3D 频谱下采样步长，默认 2")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ===== 1) 构建并加载模型 =====
    model = models.UNetHybridAttentionV23_2().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if args.weights_only:
        model.load_state_dict(ckpt, strict=True)
    else:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif isinstance(ckpt, nn.Module):
            model = ckpt.to(device)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()

    # ===== 2) 使用你给的方式加载图片 & 预处理 =====
    img_path = args.image
    image = Image.open(img_path).convert('RGB')
    # 注意：PIL 的 size 是 (W, H)
    W_orig, H_orig = image.size

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 按你的输入大小
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, 256, 256]

    with torch.no_grad():
        enhanced_tensor = model(input_tensor)  # [1, C, H', W'] (可能与 256 不同)

    # ===== 3) 还原到原图尺寸并保存增强结果 =====
    out = enhanced_tensor.clamp(0, 1)
    if out.shape[-2] != H_orig or out.shape[-1] != W_orig:
        out = F.interpolate(out, size=(H_orig, W_orig), mode="bilinear", align_corners=False)

    out_img = (out[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    enhanced_path = os.path.join(args.outdir, "enhanced.png")
    Image.fromarray(out_img).save(enhanced_path)
    print(f"[RESULT] Enhanced image saved to: {enhanced_path}")

    # ===== 4) 计算并保存 2D/3D 频谱图 =====
    F_shift, log_mag = fft2_gray(out_img)

    fft2d_path = os.path.join(args.outdir, "fft_spectrum_2d.png")
    save_fft_2d(log_mag, fft2d_path)
    print(f"[RESULT] 2D FFT spectrum saved to: {fft2d_path}")

    fft3d_path = os.path.join(args.outdir, "fft_spectrum_3d.png")
    save_fft_3d(log_mag, fft3d_path, stride=args.stride3d)
    print(f"[RESULT] 3D FFT spectrum saved to: {fft3d_path}")

    # ===== 5) 计算 LFD =====
    lfd, r_pix, r_max = compute_lfd(F_shift, radius_frac=args.radius_frac)
    print(f"[RESULT] LFD (low-freq energy ratio) = {lfd:.6f}  (radius = {r_pix:.2f} / max {r_max})")


if __name__ == "__main__":
    main()
