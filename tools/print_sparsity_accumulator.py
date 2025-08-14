"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: print_sparsity_accumulator.py
    @Time: 2025/7/29 22:23
    @Email: None
"""
import torch
from src.data.dataset import DataReader
from torch.utils.data import DataLoader
from src import models
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio



def print_sparsity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = DataReader(img_dir=r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\LSUI19\Val',
                             input='input',
                             target='target',
                             mode='test',
                             ori=False,
                             img_options={'w': 256, 'h': 256})

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True, )
    criterion_psnr = torch.nn.SmoothL1Loss()

    model = models.UNetHybridAttentionV23_2()
    state_dict = torch.load(
        r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNetHybridAttentionV23_2\20250725_013205\best_result\TOP_PSNR.pth',
        map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    psnr_total = ssim_total = 0.0
    size = len(val_loader)
    with torch.no_grad():
        for data in val_loader:
            inp, target = data[0].to(device), data[1].to(device)
            res = model(inp)
            val_loss = criterion_psnr(res, target)
            psnr = peak_signal_noise_ratio(res, target, data_range=1).item()
            psnr_total += psnr
            ssim = structural_similarity_index_measure(res, target, data_range=1).item()
            ssim_total += ssim

    psnr = psnr_total / size
    ssim = ssim_total / size
    print('psnr:', psnr)
    print('ssim:', ssim)
    all_sparsities = model.hybrid_attention1.sparsity_accumulator

    Yh_by_subband = list(zip(*all_sparsities))
    avg_sparsities = [sum(x) / len(x) for x in Yh_by_subband]

    print("每个高频子带平均稀疏性:")
    for i, s in enumerate(avg_sparsities):
        print(f"子带{i}: 稀疏性 = {s:.4f}")


if __name__ == '__main__':
    print_sparsity()
