"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: train_utils.py
    @Time: 2025/5/20 22:37
    @Email: None
"""
import os
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ExperimentLogger:
    def __init__(self, exp_root, timestamp):
        """
        :param exp_root: 实验根目录
        """
        super().__init__()
        self.exp_root = exp_root
        self.ckpt_dir = os.path.join(self.exp_root, 'checkpoints')
        self.log_dir = os.path.join(self.exp_root, 'runs')

        # timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        time_now = timestamp

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        runs_dir = os.path.join(self.log_dir, time_now)
        self.writer = SummaryWriter(log_dir=runs_dir)

    def log_metrics(self, metrics, epoch, phase='train'):
        for k, v in metrics.items():
            self.writer.add_scalar(f'{phase} / {k}', v, epoch)

    def save_checkpoint(self, model, epoch, is_best=False):
        ckpt_path = os.path.join(self.ckpt_dir, f'Epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
        }, ckpt_path)
        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)






