"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: model_output.py
    @Time: 2025/11/30 16:40
    @Email: None
"""
import torch
import torch.nn as nn
import src.models.job2_1 as job2_1

# 1. 定义超参数
B = 8  # Batch size (你可以修改为任意整数，例如 1 或 32)

# 2. 构造输入张量 (模拟输入数据)
# 形状: [Batch_Size, Channels, Height, Width] -> [B, 3, 256, 256]
input_tensor = torch.randn(B, 3, 256, 256)

print(f"输入张量的形状: {input_tensor.shape}")

model = job2_1.INN_UNetV2()

# 4. 前向传播 (Forward Pass)
# 为了测试通常不需要计算梯度，使用 torch.no_grad() 可以节省内存
model.eval()  # 切换到评估模式 (影响 Dropout, BatchNorm 等层)
with torch.no_grad():
    output = model(input_tensor)

# 5. 查看结果
print("-" * 30)
print(f"输出张量的形状: {output.shape}")
print("-" * 30)
# print(output) # 如果想看具体数值可以取消注释