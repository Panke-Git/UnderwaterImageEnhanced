"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_train_loss.py
    @Time: 2025/5/23 00:24
    @Email: None
"""
import matplotlib.pyplot as plt

import json

with open(r'/expt_record/UNet_CSC/20250524_143058/Trian.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

epochs = []
train_loss = []
val_loss = []
val_psnr = []
val_ssim  = []

for item in data:
    epochs.append(item['epoch'])
    train_loss.append(item['train_loss'])
    val_loss.append(item['val_loss'])
    val_psnr.append(item['val_psnr'])
    val_ssim.append(item['val_ssim'])

# 创建 1 行 3 列的子图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 一行三个图


ax_train_loss = axes[0, 0]
ax_val_loss = axes[0, 1]
ax_val_psnr = axes[1, 0]
ax_val_ssim = axes[1, 1]

ax_train_loss.plot(epochs, train_loss, marker='o')
ax_train_loss.set_title('train loss')
ax_val_loss.plot(epochs, val_loss, marker='o')
ax_val_loss.set_title('val loss')
ax_val_psnr.plot(epochs, val_psnr, marker='o')
ax_val_psnr.set_title('val psnr')
ax_val_ssim.plot(epochs, val_ssim, marker='o')
ax_val_ssim.set_title('val ssim')

ax_train_loss.set_ylim(min(train_loss), max(train_loss))
ax_val_loss.set_ylim(min(val_loss), max(val_loss))
ax_val_psnr.set_ylim(min(val_psnr), max(val_psnr))
ax_val_ssim.set_ylim(min(val_ssim), max(val_ssim))

plt.tight_layout()
plt.show()