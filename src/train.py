"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: train.py
    @Time: 2025/5/20 00:19
    @Email: None
"""
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from tqdm import tqdm

from src.data.dataset import DataReader
from src import models
from src.models import LargeKernel as lk
from src.utils import record_utils
from src.utils.config import Config
from src.utils.enhance_train_visual import visual_enhance, train_visual
from src.utils.train_utils import ExperimentLogger, generate_experiment_id
from src.utils.train_utils import seed_everything

import warnings

from tools.tribute_banner import show_banner

warnings.filterwarnings("ignore", message="Error fetching version info")


def train():
    config = Config.load(r'./src/config/config.yaml')
    # show_banner()
    # 开始时间
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 注册log输出
    logger = ExperimentLogger(config.PROJECT.LOG_DIR, start_time)
    # 设置随机种子
    seed_everything(3407)
    # 项目根路径，训练数据集路径，验证数据集路径
    root_path = config.PROJECT.ROOT_PATH
    train_dir = config.PROJECT.TRAIN_DIR
    val_dir = config.PROJECT.VAL_DIR
    # 训练设备
    device = torch.device(config.TRAIN.DEVICE if torch.cuda.is_available() else 'cpu')

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
                              pin_memory=True, )
    val_loader = DataLoader(val_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True, )

    # ========================================================================================
    # ==================================注意修改此值============================================
    # ========================================================================================
    model = models.UNetHybridAttention2V10().to(device)
    model_description = '使用HybridAttention2, 优化低频的部分，插入到Unet的第三层；'
    expt_id = generate_experiment_id(model=model.model_name,
                                     dataset='LSUI',
                                     loss='SmoothL1Loss',
                                     note='')
    # ========================================================================================
    # ========================================================================================
    # ========================================================================================

    epochs = config.TRAIN.EPOCHS

    criterion_psnr = torch.nn.SmoothL1Loss()

    optimizer_b = torch.optim.AdamW(model.parameters(), lr=float(config.TRAIN.LR), betas=(0.9, 0.999), eps=1e-08)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, epochs, eta_min=1e-6, last_epoch=-1)

    # 创建本次训练需要保存数据的路径；
    record_path, best_path = record_utils.make_