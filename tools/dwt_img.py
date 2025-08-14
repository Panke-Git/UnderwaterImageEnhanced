"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: dwt_img.py
    @Time: 2025/7/26 00:10
    @Email: None
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import pywt
import matplotlib.pyplot as plt

# Step 1: 读取图像并转为张量
image_bgr = cv2.imread('2648.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 转为 PyTorch tensor, [C, H, W], 并归一化到 [0,1]
image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # shape: [1, 3, H, W]

# Step 2: 定义一个普通 3x3 卷积（你也可以改成别的核/通道数）
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
# 可选：手动初始化为 edge detection / identity / blur 等，也可以随机
nn.init.xavier_uniform_(conv.weight)  # 使用 Xavier 初始化

# 进行卷积
with torch.no_grad():
    conv_output = conv(image_tensor)  # shape: [1, 3, H, W]
conv_output_np = conv_output.squeeze(0).permute(1, 2, 0).numpy()  # shape: [H, W, 3]

# Step 3: 对每个通道做 DWT
def dwt_rgb_channels(image_np):
    cA_list, cH_list, cV_list, cD_list = [], [], [], []
    for i in range(3):  # R/G/B 三个通道
        cA, (cH, cV, cD) = pywt.dwt2(image_np[:, :, i], 'haar')
        cA_list.append(cA)
        cH_list.append(cH)
        cV_list.append(cV)
        cD_list.append(cD)

    def stack_normalize(channels):
        out = np.stack(channels, axis=-1)
        out = out - out.min()
        out = out / out.max()
        return (out * 255).astype(np.uint8)

    return (
        stack_normalize(cA_list),
        stack_normalize(cH_list),
        stack_normalize(cV_list),
        stack_normalize(cD_list),
    )

# 执行 DWT
cA_rgb, cH_rgb, cV_rgb, cD_rgb = dwt_rgb_channels(conv_output_np)

# Step 4: 分别显示四张图
def show(img, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

show(cA_rgb, 'Approximation (cA)')
show(cH_rgb, 'Horizontal Detail (cH)')
show(cV_rgb, 'Vertical Detail (cV)')
show(cD_rgb, 'Diagonal Detail (cD)')
