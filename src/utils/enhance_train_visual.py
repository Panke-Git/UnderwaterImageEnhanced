"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: visual.py
    @Time: 2025/5/24 17:23
    @Email: None
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.dataset import DataReader
# from models.Unet_CSC import UNet_CSC


def tensor_to_img(tensor):
    """Tensor(C,H,W) -> Numpy(H,W,C)，范围[0,1]"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def visual_enhance(model_in=None, root_dir='', img=''):
    test_dataset = DataReader(img_dir=img,
                              input='input',
                              target='GT',
                              mode='test',
                              ori=False,
                              img_options={'w': 256, 'h': 256})
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    img_count = len(test_dataloader)
    device = 'cpu'
    inputs_list = []
    target_list = []
    output_list = []
    model = model_in.to(device=device)
    pth_path = os.path.join(root_dir, 'best_result', 'TOP_PSNR.pth')
    pth_load = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(pth_load)
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            inp, tar = data[0].to(device), data[1].to(device)
            res = model(inp)
            # print(inputs[0].shape, targets[0].shape, result[0].shape)

            inputs_list.append(tensor_to_img(inp[0]))
            target_list.append(tensor_to_img(tar[0]))
            output_list.append(tensor_to_img(res[0]))

    fig, axs = plt.subplots(img_count, 3, figsize=(10, 3 * int(img_count)))

    for i in range(img_count):
        axs[i, 0].imshow(inputs_list[i])
        axs[i, 0].set_title("Input")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(output_list[i])
        axs[i, 1].set_title("Enhanced")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(target_list[i])
        axs[i, 2].set_title("Ground Truth")
        axs[i, 2].axis("off")

    # plt.suptitle(f'Visual Enhancement IMG of {model.model_name}', fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    # save_path = os.path.join(root_dir, 'visual_enhance.png')
    # plt.savefig(save_path)
    # plt.close()


def train_visual(root_dir=''):
    train_data_path = os.path.join(root_dir,'Trian.json')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    epochs = []
    train_loss = []
    val_loss = []
    val_psnr = []
    val_ssim = []

    for item in train_data:
        epochs.append(item['epoch'])
        train_loss.append(item['train_loss'])
        val_loss.append(item['val_loss'])
        val_psnr.append(item['val_psnr'])
        val_ssim.append(item['val_ssim'])

    # 创建 1 行 3 列的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

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

    plt.suptitle(f'Training parameter visualization', fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    save_path = os.path.join(root_dir, 'visual_training.png')
    plt.savefig(save_path)
    plt.close()
