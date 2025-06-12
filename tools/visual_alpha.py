# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： visual_alpha.py
    @Date：2025/6/12 11:01
    @Email: None
"""
import json

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

json_file = r'./Trian.json'
with open(json_file) as f:
    data = json.load(f)
alpha1s = []
alpha2s = []
for item in data:
    alpha1s.append(item['alpha1'])
    alpha2s.append(item['alpha2'])

alpha1s = np.array(alpha1s)
alpha2s = np.array(alpha2s)

plt.figure(figsize=(14, 8))
sns.heatmap(alpha1s.T, cmap="viridis", cbar=True, xticklabels=50, yticklabels=16)
plt.xlabel("Epoch")
plt.ylabel("Channel Index")
plt.title("Alpha values heatmap (channel vs. epoch)")
plt.tight_layout()
plt.show()
alpha_mean = alpha1s.mean(axis=1)  # 每一轮的 alpha 平均值

plt.plot(alpha_mean)
plt.xlabel("Epoch")
plt.ylabel("Mean alpha value")
plt.title("Mean alpha over epochs")
plt.grid(True)
plt.tight_layout()
plt.show()