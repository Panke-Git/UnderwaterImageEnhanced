"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_pixel_value.py
    @Time: 2025/10/28 22:43
    @Email: None
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def y_from_bgr(img, standard="bt601"):
    # 提取 Y（luma, Y'）通道
    if standard.lower() == "bt601":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    elif standard.lower() == "bt709":
        b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)
    else:
        raise ValueError("standard 只能是 'bt601' 或 'bt709'")

def plot_two_y_hist(
    img_path1, img_path2,
    standard="bt601",      # 'bt601' 或 'bt709'
    range_mode="full",     # 8-bit: 'full' = 0..255；'video' = 16..235
    bit_depth=8,           # 10/12-bit 图像可改为 10/12；uint16 数据会被当做该位深
    y_scale=1e-4,          # 纵轴缩放因子；默认把计数除以 1e4 以便显示
    label_scale="×10⁻⁴",   # y 轴单位后缀（配合 y_scale 显示）
    save_path=None
):
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError("无法读取图像，请检查路径。")

    Y1 = y_from_bgr(img1, standard)
    Y2 = y_from_bgr(img2, standard)

    # 统一直方图的 bins 与取值范围，保证两条曲线可对齐比较
    if bit_depth == 8:
        if range_mode == "full":
            bins, hist_range = 256, (0, 256)
        elif range_mode == "video":
            bins, hist_range = 235 - 16 + 1, (16, 236)
        else:
            raise ValueError("range_mode 只能是 'full' 或 'video'")
    else:
        bins = 2 ** bit_depth
        hist_range = (0, 2 ** bit_depth)

    counts1, edges = np.histogram(Y1, bins=bins, range=hist_range)
    counts2, _     = np.histogram(Y2, bins=bins, range=hist_range)
    x = edges[:-1].astype(int)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(x, counts1 * y_scale, linewidth=1, label='input')
    plt.plot(x, counts2 * y_scale, linewidth=1, label='GT')
    plt.xlabel("Pixel Value")
    plt.ylabel(f"Pixel Number {label_scale}")
    plt.title("Y Channel Histogram (Two Images)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# 用法示例：
# plot_two_y_hist("a.jpg", "b.jpg", standard="bt601", range_mode="full")
# 如果你“真的想把纵轴放大 10^4 倍”，用：
# plot_two_y_hist("a.jpg", "b.jpg", y_scale=1e4, label_scale="×10^4")

# 用法示例：
# plot_y_histogram("your_image.png", standard="bt601", range_mode="full")1368_256_out.jpg
input_img = r'F:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\1368\1368_input256.jpg'
output_img = r'F:\01_Storage Area\CLFile\01_Doc\img\LSUI\V23_2\1368\1368_256_out.jpg'

plot_two_y_hist(input_img, output_img, standard="bt601", range_mode="full", save_path="y_hist.png")
