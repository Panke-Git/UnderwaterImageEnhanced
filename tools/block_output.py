"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: block_output.py
    @Time: 2025/5/24 11:09
    @Email: None
"""

# from src import models
import src.models.job1_1 as job1_1
import torch
# from src.utils.enhance_train_visual import visual_enhance


x = torch.randn(1, 3, 256, 256).to(torch.device('cuda:0'))

# block = CSC_block(3)
unet = job1_1.UNetHybridAttentionV31().to(torch.device('cuda:0'))

# net = lk.UNetCSC_LKA_SDCA_FDPA().to(torch.device('cuda:0'))


# print(x.shape)
out = unet(x)
print(out.shape)

# img_path = r'C:\Users\Panke\Desktop\UIR_IMG'
# model= models.UNetHybridAttentionV23()
# visual_enhance(model, r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNetHybridAttentionV23\20250610_000143', img_path)





