"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: evaluate_metrics_from_pth.py
    @Time: 2025/6/24 23:45
    @Email: None
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src import models
# from dataset import get_val_loader  # 你自己定义的 DataLoader
from UnderwaterMetrics import UnderwaterMetrics
from src.data.dataset import DataReader
from torch.utils.data import DataLoader

from piq import psnr, ssim  # 可选

# or from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
# from torchmetrics.functional.image import structural_similarity_index_measure as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载模型
model = models.UNetHybridAttentionV23().to(device)
model.load_state_dict(torch.load(
    r"E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNetHybridAttentionV23\20250610_000143\best_result\TOP_PSNR.pth",
    map_location=device))
model.eval()

# 2. 加载验证数据
# val_loader = get_val_loader(batch_size=1, shuffle=False)  # 每次处理一张，方便计算评价指标
val_dataset = DataReader(img_dir=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUIN\Val',
                         input='input',
                         target='GT',
                         mode='test',
                         ori=False,
                         img_options={'w': 256, 'h': 256})
val_loader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        # num_workers=4,
                        pin_memory=True, )

# 3. 评价器
metric_calc = UnderwaterMetrics()

# 4. 遍历验证集
uiqm_sum = uciqe_sum = fdum_sum = ccf_sum = pcqi_sum = cbpd_sum = 0.0
psnr_sum = ssim_sum = 0.0

with torch.no_grad():
    for _, data in enumerate(tqdm(val_loader)):
        inp, target = data[0].to(device), data[1].to(device)
        # inp = inp.to(device)  # [1, 3, H, W]
        # GT = GT.to(device)  # [1, 3, H, W]
        pred = model(inp)  # 你的模型输出，应该 shape 和 inp 一致
        # print(pred.shape)

        # 去除 batch 维度
        pred_img = pred[0].clamp(0, 1)
        target_img = target[0].clamp(0, 1)

        # PSNR 和 SSIM
        psnr_val = psnr(pred_img.unsqueeze(0), target_img.unsqueeze(0), data_range=1.0).item()
        ssim_val = ssim(pred_img.unsqueeze(0), target_img.unsqueeze(0), data_range=1.0).item()

        psnr_sum += psnr_val
        ssim_sum += ssim_val

        # 计算水下图像评价指标
        metrics = metric_calc.compute_metrics(pred_img, target_img)
        uiqm_sum += metrics['UIQM']
        uciqe_sum += metrics['UCIQE']
        fdum_sum += metrics['FDUM']
        ccf_sum += metrics['CCF']
        pcqi_sum += metrics['PCQI']
        cbpd_sum += metrics['CBPD']

# 5. 汇总
n = len(val_loader)
results = {
    "PSNR": psnr_sum / n,
    "SSIM": ssim_sum / n,
    "UIQM": uiqm_sum / n,
    "UCIQE": uciqe_sum / n,
    "FDUM": fdum_sum / n,
    "CCF": ccf_sum / n,
    "PCQI": pcqi_sum / n,
    "CBPD": cbpd_sum / n
}

# 6. 打印 & 保存
print(json.dumps(results, indent=2))

with open("eval_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
