"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_3DFFT.py
    @Time: 2025/9/9 21:44
    @Email: None
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import src.models as models

from torchvision import transforms


# ========== 1) 按你的项目修改这里 ==========
def build_model():
    """
    TODO: 按你的项目返回模型实例（未加载权重）。
    例如：
        from models.my_net import MyNet
        return MyNet()
    """

    # 示例：一个占位的 3x3 卷积网络（请改成你的真实模型）
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return torch.sigmoid(self.conv(x))

    return DummyNet()


# ========================================


def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # 转为 [0,1] float32 CHW
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return x, (h, w)


def tensor_to_image(t):
    # t: 1,C,H,W in [0,1]
    t = t.clamp(0, 1).detach().cpu()[0]  # C,H,W
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def run_inference(model, x, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y = model(x)
    return y.cpu()


def save_fft_spectrum(img_arr, save_path):
    """
    img_arr: numpy uint8 HxWx3 或 HxW（0~255）
    绘制对数幅度频谱（中心化），并保存。
    """
    # 转灰度以计算频谱（也可对每通道分别做；这里简化为灰度）
    if img_arr.ndim == 3:
        gray = np.dot(img_arr[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = img_arr.astype(np.float32)

    gray = gray.astype(np.float32)
    # 归一化到 [0,1]
    if gray.max() > 1.0:
        gray = gray / 255.0

    # 计算 2D FFT
    F_uv = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F_uv)
    mag = np.abs(F_shift)

    # 对数幅度（+1 防止 log(0)）
    log_mag = np.log1p(mag)

    # 归一化到 [0,1] 便于显示
    log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-12)

    plt.figure(figsize=(6, 6))
    plt.imshow(log_mag_norm, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()

    return F_shift  # 供 LFD 计算复用


def compute_lfd(F_shift, radius_frac=0.10):
    """
    F_shift: 已经 fftshift 的频谱（复数矩阵）
    LFD 定义为：低频能量 / 总能量
    radius_frac: 半径比例（0~1），相对最大半径（以最小的半径方向为准）
    """
    H, W = F_shift.shape
    cy, cx = H // 2, W // 2
    # 能量谱
    power = np.abs(F_shift) ** 2

    # 以图像中心为圆心构造半径网格
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    max_r = min(cy, cx)  # 频谱最大可用半径
    r = radius_frac * max_r

    low_mask = dist <= r
    low_energy = power[low_mask].sum()
    total_energy = power.sum() + 1e-12

    return float(low_energy / total_energy), float(r), int(max_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record_HNUST\UNetHybridAttentionV23_2\20250819_222858\best_result\TOP_PSNR.pth',
                        help="模型权重 .pth 路径（state_dict 或整模型）")
    parser.add_argument("--image", type=str, default=r'./img/input.jpg', help="输入图片路径")
    parser.add_argument("--outdir", type=str, default="./out")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights-only", action="store_true",
                        help="若 ckpt 保存的是 state_dict，请加此参数")
    parser.add_argument("--radius-frac", type=float, default=0.10,
                        help="LFD 低频半径比例（相对最大半径），如 0.10 表示 10%%")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 构建并加载模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = models.UNetHybridAttentionV23_2().to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if args.weights_only:
        # 你的 .pth 是 state_dict
        model.load_state_dict(ckpt, strict=True)
    else:
        # 你的 .pth 是整模型（或包含 'state_dict'）
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif isinstance(ckpt, nn.Module):
            model = ckpt.to(device)
        else:
            # 尝试直接当 state_dict
            model.load_state_dict(ckpt, strict=False)

    # 2) 读取图片并推理
    image = Image.open(args.image).convert('RGB')
    H_orig, W_orig  = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 视你的模型输入大小而定
        transforms.ToTensor(),
    ])

    # 与输入尺寸一致（若你的网络改变了尺寸，这里强制恢复原始尺寸）
    if y.shape[-2] != h or y.shape[-1] != w:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

    # 保存增强后图像
    out_img = tensor_to_image(y)
    out_img_path = os.path.join(args.outdir, "enhanced.png")
    out_img.save(out_img_path)

    # 3) 频谱图 + LFD
    out_np = np.array(out_img)  # HxWx3
    fft_img_path = os.path.join(args.outdir, "fft_spectrum.png")
    F_shift = save_fft_spectrum(out_np, fft_img_path)
    lfd, r_pix, r_max = compute_lfd(F_shift, radius_frac=args.radius_frac)

    print(f"[RESULT] Enhanced image saved to: {out_img_path}")
    print(f"[RESULT] FFT spectrum saved to : {fft_img_path}")
    print(f"[RESULT] LFD (low-freq energy ratio) = {lfd:.6f}  (radius = {r_pix:.2f} / max {r_max})")


if __name__ == "__main__":
    main()
