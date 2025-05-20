"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_img.py
    @Time: 2025/5/20 21:47
    @Email: None
"""
import os

import torch


def tensor_to_img(tensor):
    if tensor.ndim == 4:
        tensor = tensor.sequeeze(0)
    tensor = tensor.detach().cpu().clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def evaluate_on_validation_set(model, dataloader, device, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    count = 0
    num_samples = 5

    inputs_list, outputs_list, targets_list = [], [], []

    with torch.no_grad():
        for data in dataloader:
            inp, tar = data[0].to(device), data[1].to(device)
            res = model(inp)

            for i in range(inp.shape[0]):
                inputs_list.append(inp[i])
                outputs_list.append(res[i])
                targets_list.append(tar[i])

                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        axs[i, 0].imshow(inputs_list[i])
        axs[i, 1].set_title('Input')
        axs[i, 2].axis('off')

        axs[i, 1].imgshow(outputs_list[i])
        axs[i, 1].set_title('Enhanced')
        axs[i, 1].axis('off')

        axs[i, 2].imgshow(targets_list[i])
        axs[i, 2].set_title('Target')
        axs[i, 2].axis('off')

    fig.suptitle(f'Epoch {epoch + 1} - Visual Results', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    save_path = os.path.join(save_dir, f'epoch_{epoch + 1}_visual.png')
    plt.savefig(save_path)
    plt.close()
