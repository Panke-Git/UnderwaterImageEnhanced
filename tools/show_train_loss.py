"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: show_train_loss.py
    @Time: 2025/5/23 00:24
    @Email: None
"""
import matplotlib.pyplot as plt

import json

with open(r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\expt_record\UNet\20250522_223342\Trian__20250522_223342——20250523_001738.json', 'r', encoding='utf-8') as f:
    data = json.load(f)





#
# # 创建 1 行 3 列的子图
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 一行三个图
#
# for i, y_data in enumerate(data_list):
#     x_data = list(range(1, len(y_data) + 1))
#     ax = axes[i]
#     ax.plot(x_data, y_data, marker='o')
#     ax.set_title(f"Subplot {i + 1}")
#     ax.set_ylim(min(y_data) - 1, max(y_data) + 1)
#     ax.grid(True)
#
# plt.tight_layout()
# plt.show()