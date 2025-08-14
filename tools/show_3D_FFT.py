"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_3D_FFT.py
    @Time: 2025/7/28 22:30
    @Email: None
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src import models
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def tensor_to_img(tensor):
    """Tensor(C,H,W) -> Numpy(H,W,C)，范围[0,1]"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def visual_enhance():
    model = models.UNetHybridAttentionV23_2()
    model.load_state_dict(torch.load(
        r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNetHybridAttentionV23_2\20250725_013205\best_result\TOP_PSNR.pth',
        map_location='cuda'))  # 替换路径
    model.eval()

    # -------------------
    # 2. 加载并预处理图片
    # -------------------
    img_path = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\tools\img\input.jpg'  # 替换为你自己的图片路径
    image = Image.open(img_path).convert('RGB')
    H_orig, W_orig  = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 视你的模型输入大小而定
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度

    # -------------------
    # 3. 前向推理得到增强图像
    # -------------------
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)

    # 5. 将输出tensor resize回原始图像大小
    output_tensor_resized = F.interpolate(enhanced_tensor, size=(W_orig, H_orig), mode='bilinear', align_corners=False)

    # 6. 转为numpy图像
    output_img = output_tensor_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 7. 反归一化，转uint8（根据你的数据范围调整）
    output_img = (output_img * 255).clip(0, 255).astype('uint8')
    # 反归一化，确保值在0~1之间
    # output_img = np.clip(output_img, 0, 1)

    # plt展示
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(output_img)
    plt.show()

# visual_enhance()

def show_3d_FLY():
    # 1. 读取图像（灰度）
    img_name = r'img/input.jpg'
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # 可选：图像缩小避免渲染过慢

    # 2. 傅里叶变换（频域分析）
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # 3. 构造 X, Y 网格坐标
    X = np.arange(magnitude_spectrum.shape[1])
    Y = np.arange(magnitude_spectrum.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = magnitude_spectrum

    # 4. 绘制 3D 表面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(img_name.split('/')[1])

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()



show_3d_FLY()

