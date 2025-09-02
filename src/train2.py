"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: train.py
    @Time: 2025/5/20 00:19
    @Email: None
"""
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from data.dataset import DataReader
import models
from models import LargeKernel as lk
from utils import record_utils
from utils.config import Config
from utils.enhance_train_visual import visual_enhance, train_visual
from utils.train_utils import ExperimentLogger, generate_experiment_id
from utils.train_utils import seed_everything

import warnings

# from tools.tribute_banner import show_banner

warnings.filterwarnings("ignore", message="Error fetching version info")


def train():
    time.sleep(1)
    config = Config.load(r'/public/home/hnust15874739861/pro/UnderwaterImageEnhanced/src/config/config2.yaml')
    # show_banner()
    # 开始时间
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 注册log输出
    logger = ExperimentLogger(config.PROJECT.LOG_DIR, start_time)
    # 设置随机种子
    seed_everything(9861)
    # 项目根路径，训练数据集路径，验证数据集路径
    root_path = config.PROJECT.ROOT_PATH
    train_dir = config.PROJECT.TRAIN_DIR
    val_dir = config.PROJECT.VAL_DIR
    # 训练设备
    device = torch.device(config.TRAIN.DEVICE if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    train_dataset = DataReader(img_dir=train_dir,
                               input=config.DATASET.INPUT,
                               target=config.DATASET.TARGET,
                               mode='train',
                               ori=True,
                               img_options={'w': config.TRAIN.IMG_W, 'h': config.TRAIN.IMG_H})
    val_dataset = DataReader(img_dir=val_dir,
                             input=config.DATASET.INPUT,
                             target=config.DATASET.TARGET,
                             mode='test',
                             ori=False,
                             img_options={'w': config.TRAIN.IMG_W, 'h': config.TRAIN.IMG_H})

    train_loader = DataLoader(train_dataset,
                              batch_size=config.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=use_cuda, )
    val_loader = DataLoader(val_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=use_cuda, )

    # ========================================================================================
    # ==================================注意修改此值============================================
    # ========================================================================================
    model = models.UNetHybridAttentionV23_2().to(device)
    model_description = '基于UNetHybridAttentionV23，将HybridAttention前边加入卷积；'
    model_name = model.model_name
    # model_name = 'UNetHybridAttentionV23_2'
    expt_id = generate_experiment_id(model=model_name,
                                     dataset='LSUI19',
                                     loss='SmoothL1Loss',
                                     note='')
    # ========================================================================================
    # ========================================================================================
    # ========================================================================================

    epochs = config.TRAIN.EPOCHS

    criterion_smoothl1  = torch.nn.SmoothL1Loss()

    optimizer_b = torch.optim.AdamW(model.parameters(), lr=float(config.TRAIN.LR), betas=(0.9, 0.999), eps=1e-08)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, epochs, eta_min=1e-6, last_epoch=-1)

    # 创建本次训练需要保存数据的路径；
    record_path, best_path = record_utils.make_train_path(config.PROJECT.EXPT_RECORD_DIR, model_name, start_time)

    config_file_path = record_utils.save_train_config(record_path,
                                                      model=model_name,
                                                      expt_id=expt_id,
                                                      model_description=model_description,
                                                      batch_size=config.TRAIN.BATCH_SIZE,
                                                      lr=float(config.TRAIN.LR),
                                                      epochs=epochs,
                                                      scheduler=str(scheduler_b),
                                                      optimizer=str(optimizer_b),
                                                      dataset=train_dir,
                                                      seed_value=9861
                                                      )
    record_utils.record_model_description(
        os.path.join(config.PROJECT.ROOT_PATH, 'src', 'models', '01_ModelDescription.json'),
        model_name=model_name, model_description=model_description)
    print("配置信息保存至: ", config_file_path, "下!")
    tdr = train_dir.split('/')[-2]
    print(f'本次训练配置: epoch: {epochs}, dataset: {tdr}, batch_size: {config.TRAIN.BATCH_SIZE}')

    top_psnr = 0.0
    top_ssim = 0.0
    sum_psnr_ssim = 0.0
    top_data = None
    total_record = []
    top_psnr_data = {
        'epoch': 0,
        'train_loss': 0,
        'val_loss': 0,
        'psnr': 0,
        'ssim': 0,
    }
    top_ssim_data = top_psnr_data.copy()
    top_sum_data = top_psnr_data.copy()

    size = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        num_train_batches = 0
        for _, data in enumerate(tqdm(train_loader)):
            inp, target = data[0].to(device), data[1].to(device)

            optimizer_b.zero_grad()

            res = model(inp)
            res_clamped = res.clamp(0, 1)

            loss_reg = criterion_smoothl1(res, target)
            ssim_val = structural_similarity_index_measure(res_clamped, target, data_range=1)
            loss_ssim = 1 - ssim_val

            train_loss = loss_reg + 0.2 * loss_ssim
            train_loss.backward()
            optimizer_b.step()

            train_loss_total += train_loss.item()
            num_train_batches += 1

        epoch_train_loss = train_loss_total / max(1, num_train_batches)
        scheduler_b.step()
        logger.writer.add_scalar('train/loss', epoch_train_loss, epoch)

        if epoch % config.TRAIN.PRINT_FREQ == 0:
            model.eval()

            val_loss_total = 0.0
            psnr_total = 0.0
            ssim_total = 0.0
            num_images = 0
            with torch.inference_mode():
                for data in tqdm(val_loader):
                    inp, target = data[0].to(device), data[1].to(device)
                    res = model(inp).clamp(0, 1)

                    batch_loss = criterion_smoothl1(res, target)
                    val_loss_total += batch_loss.item()

                    B = res.size(0)
                    # psnr_batch = peak_signal_noise_ratio(res, GT, data_range=1)
                    psnr_each = peak_signal_noise_ratio(res, target, data_range=1.0, dim=(1, 2, 3), reduction='none')
                    ssim_batch = structural_similarity_index_measure(res, target, data_range=1)

                    # psnr_total += psnr_batch.item() * B
                    psnr_total += psnr_each.sum().item()
                    ssim_total += ssim_batch.item() * B
                    num_images += B

            # ==== 修改点 2-4：正确的平均 ====
            psnr = psnr_total / max(1, num_images)
            ssim = ssim_total / max(1, num_images)
            val_loss_avg = val_loss_total / max(1, len(val_loader))

            metrics = {'PSNR': psnr, 'SSIM': ssim}

            epoch_record = record_utils.package_one_epoch(epoch=epoch,
                                                          train_loss=float(epoch_train_loss),
                                                          val_loss=float(val_loss_avg),
                                                          val_psnr=float(psnr),
                                                          val_ssim=float(ssim),
                                                          lr=float(optimizer_b.param_groups[0]["lr"]))
            total_record.append(epoch_record)

            logger.log_metrics({'loss': val_loss_avg}, epoch, 'val')
            logger.log_metrics({'psnr': psnr}, epoch, 'val')
            logger.log_metrics({'ssim': ssim}, epoch, 'val')

            if metrics['PSNR'] > top_psnr:
                top_psnr = metrics['PSNR']
                top_psnr_path = os.path.join(best_path, f'TOP_PSNR.pth')
                top_psnr_data = {
                    'epoch': epoch,
                    'train_loss': float(epoch_train_loss),
                    'val_loss': float(val_loss_avg),
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                }
                torch.save(model.state_dict(), top_psnr_path)

            if metrics['SSIM'] > top_ssim:
                top_ssim = metrics['SSIM']
                top_ssim_path = os.path.join(best_path, f'TOP_SSIM.pth')
                top_ssim_data = {
                    'epoch': epoch,
                    'train_loss': float(epoch_train_loss),
                    'val_loss': float(val_loss_avg),
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                }
                torch.save(model.state_dict(), top_ssim_path)

            if metrics['PSNR'] + metrics['SSIM'] * 100 > sum_psnr_ssim:
                sum_psnr_ssim = metrics['PSNR'] + metrics['SSIM'] * 100
                sum_path = os.path.join(best_path, f'TOP_SUM.pth')
                top_sum_data = {
                    'epoch': epoch,
                    'train_loss': float(epoch_train_loss),
                    'val_loss': float(val_loss_avg),
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                }
                torch.save(model.state_dict(), sum_path)

            top_data = {
                'PSNR': {
                    'top_psnr': float(top_psnr),
                    'top_psnr_data': top_psnr_data,
                },
                'SSIM': {
                    'top_ssim': float(top_ssim),
                    'top_ssim_data': top_ssim_data
                },
                'SUM': {
                    'top_sum': float(sum_psnr_ssim),
                    'top_sum_data': top_sum_data
                }
            }

            print(f'epoch: {epoch}/{epochs}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f};\n'
                  f"Best PSNR: {top_psnr_data['psnr']:.4f}, Best PSNR_epoch: {top_psnr_data['epoch']};\n"
                  f"Best SSIM: {top_ssim_data['ssim']:.4f}, Best SSIM_epoch: {top_ssim_data['epoch']};\n"
                  f'LR: {optimizer_b.param_groups[0]["lr"]:.6f}')

    end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path, json_path, top_path = record_utils.save_train_data(record_path, start_time, end_time, total_record,
                                                                   top_data)
    print(f'数据已保存: \n \t Excel: {excel_path} \n \t Json: {json_path} \n \t Top: {top_path}')

    print("正在可视化结果，请稍等...🤖")
    visual_enhance(model, record_path, config.PROJECT.VISUAL_DATA)
    train_visual(record_path)
    print("可视化完成!✌️")


if __name__ == '__main__':
    train()
