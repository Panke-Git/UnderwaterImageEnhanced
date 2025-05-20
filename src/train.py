"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: train.py
    @Time: 2025/5/20 00:19
    @Email: None
"""
import os

import torch
from mmengine.runner import save_checkpoint
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from tqdm import tqdm

from src.utils.train_utils import seed_everything
from src.utils.config import Config
from src.data.dataset import DataReader
from src.models.Unet import UNet


def train():
    config = Config.load('config.yaml')
    seed_everything()
    root_path = config.PROJECT.ROOT_PATH
    train_dir = os.path.join(root_path, config.PROJECT.TRAIN_DIR)
    val_dir = os.path.join(root_path, config.PROJECT.VAL_DIR)
    device = torch.device(config.TRAIN.DEVICE if torch.cuda.is_available() else 'cpu')
    train_dataset = DataReader(img_dir=train_dir,
                               input=config.DATASET.INPUT,
                               target=config.DATASET.TARGET,
                               model='train',
                               ori=True,
                               img_options={'w': config.TRAIN.IMG_W, 'h': config.TRAIN.IMG_H})
    val_dataset = DataReader(img_dir=val_dir,
                             input=config.DATASET.INPUT,
                             target=config.DATASET.TARGET,
                             model='test',
                             ori=True,
                             img_options={'w': config.TRAIN.IMG_W, 'h': config.TRAIN.IMG_H})

    train_loader = DataLoader(train_dataset,
                              batch_size=config.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True, )
    val_loader = DataLoader(val_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True, )
    model = UNet().to(device)
    epochs = config.TRAIN.EPOCHS

    criterion_psnr = torch.nn.SmoothL1Loss()

    optimizer_b = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, epochs, eta_min=1e-6, last_epoch=-1)

    best_psnr_epoch = 1
    best_psnr = 0

    size = len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()

        for _, data in enumerate(tqdm(train_loader)):
            inp, target = data[0].to(device), data[1].to(device)

            optimizer_b.zero_grad()
            res = model(inp)
            loss_psnr = criterion_psnr(res, target)
            ssim_val = structural_similarity_index_measure(res, target, data_range=1)

            loss_ssim = 1 - ssim_val
            train_loss = loss_psnr + loss_ssim * 0.2

            train_loss.backward()
            optimizer_b.step()
        scheduler_b.step()

        if epoch % config.TRAIN.PRINT_FREQ == 0:
            model.eval()
            psnr_total = ssim_total = 0.0
            size = len(val_loader)
            with torch.no_grad():
                for data in tqdm(val_loader):
                    inp, target = data[0].to(device), data[1].to(device)
                    res = model(inp)
                    psnr_total += peak_signal_noise_ratio(res, target, data_range=1).item()
                    ssim_total += structural_similarity_index_measure(res, target, data_range=1).item()
            psnr = psnr_total / size
            ssim = ssim_total / size

            if psnr > best_psnr:
                best_psnr = psnr
                best_psnr_epoch = epoch
                save_checkpoint()
            print(f'epoch: {epoch}/{epochs}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f},'
                  f'Best PSNR: {best_psnr:.4f}, Best PSNR_epoch: {best_psnr_epoch}'
                  f'LR: {optimizer_b.param_groups[0]["lr"]:.4f}')


if __name__ == '__main__':
    train()
