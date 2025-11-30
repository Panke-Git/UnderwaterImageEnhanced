"""
    @Project: UnderwaterImageEnhanced
    @Author: paxton
    @FileName： SIREN.py
    @Date：2025/11/29 00:33
    @OS：
    @Email: None
"""
# siren.py
# 完整的 SIREN (Sinusoidal Representation Network) 实现：
# - Sine 激活
# - SirenLayer：带论文初始化的单层
# - SirenNet：多层 SIREN MLP
# - create_coord_grid：方便为 1D/2D/3D 场景构造坐标网格

import math
import torch
import torch.nn as nn


class Sine(nn.Module):
    """
    y = sin(w0 * x)
    SIREN 中使用的周期激活函数。
    """

    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """
    单层 SIREN：
        y = sin( w0 * (W x + b) )

    参数：
        in_features   输入维度
        out_features  输出维度
        w0            该层的频率系数
        c             初始化中的常数，论文中通常取 6
        is_first      是否为网络的首层（首层和后续层初始化不同）
        bias          是否使用偏置
    """

    def __init__(self,
                 in_features,
                 out_features,
                 w0=1.0,
                 c=6.0,
                 is_first=False,
                 bias=True):
        super(SirenLayer, self).__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.c = c

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        # 根据论文给出的建议初始化方式：
        # 首层：      W ~ U(-1/in, 1/in)
        # 隐藏层：    W ~ U(-sqrt(c/in)/w0, sqrt(c/in)/w0)
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(self.c / self.in_features) / self.w0

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                # 常见做法是将 bias 初始化为 0
                self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SirenNet(nn.Module):
    """
    多层 SIREN 网络，用于隐式表示（implicit neural representation）。

    典型用法：
        - 输入：坐标或“坐标 + 其他特征”，维度 = in_features
        - 输出：RGB / 标量场 / 特征向量，维度 = out_features

    参数：
        in_features     输入维度（例如 2 对应 (x,y)，3 对应 (x,y,z)，
                         或者 "特征 + 坐标" 的总维度）
        hidden_features 隐藏层宽度
        hidden_layers   隐藏层层数（不含首层和最后一层线性层）
        out_features    输出维度
        w0              隐藏层的频率系数
        w0_initial      首层的频率系数，一般设得更大（如 30）
        c               初始化中的常数（论文建议 6）
        outermost_linear 是否让最后一层保持线性（不加 sin）
        final_activation 最后一层之后的激活函数（如 nn.Sigmoid()），
                         如果不需要可以设为 None
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 w0=1.0,
                 w0_initial=30.0,
                 c=6.0,
                 outermost_linear=True,
                 final_activation=None,
                 bias=True):
        super(SirenNet, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.w0 = w0
        self.w0_initial = w0_initial
        self.c = c
        self.outermost_linear = outermost_linear
        self.final_activation = final_activation

        layers = []

        # 首层：使用 w0_initial，并标记 is_first=True
        first_layer = SirenLayer(
            in_features=in_features,
            out_features=hidden_features,
            w0=w0_initial,
            c=c,
            is_first=True,
            bias=bias
        )
        layers.append(first_layer)

        # 隐藏层：使用统一的 w0
        for _ in range(hidden_layers):
            layers.append(
                SirenLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    w0=w0,
                    c=c,
                    is_first=False,
                    bias=bias
                )
            )

        self.net = nn.ModuleList(layers)

        # 最后一层：通常是线性层（论文中最后一层一般不用 sin）
        if outermost_linear:
            self.final_layer = nn.Linear(hidden_features, out_features, bias=bias)
            self.init_final_layer()
        else:
            # 如需最后一层也用 SIREN，可以换成 SirenLayer
            self.final_layer = SirenLayer(
                in_features=hidden_features,
                out_features=out_features,
                w0=w0,
                c=c,
                is_first=False,
                bias=bias
            )

    def init_final_layer(self):
        # 最后一层线性层的初始化，可以与隐藏层相同的规则
        with torch.no_grad():
            in_features = self.final_layer.in_features
            bound = math.sqrt(self.c / in_features) / self.w0
            self.final_layer.weight.uniform_(-bound, bound)
            if self.final_layer.bias is not None:
                self.final_layer.bias.zero_()

    def forward(self, x):
        """
        x: 形状 (..., in_features)，可以是任意维度，只要最后一维是特征维
        返回：同样形状的 (..., out_features)
        """
        # 按论文做法：所有中间层用 sin，最后一层线性
        for layer in self.net:
            x = layer(x)

        x = self.final_layer(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


def create_coord_grid(n_coords, dim, ranges=None, device=None):
    """
    创建 1D / 2D / 3D 等场景的坐标网格，用于配合 SIREN 使用。

    例如：
        - 2D 图像：dim=2，n_coords=(H, W)，返回 (H*W, 2)
        - 1D 曲线：dim=1，n_coords=(N,)，返回 (N, 1)
        - 3D 体数据：dim=3，n_coords=(D, H, W)，返回 (D*H*W, 3)

    参数：
        n_coords:  tuple 或 list，例如 (H, W) / (D, H, W)
        dim:       坐标维度（1/2/3...）
        ranges:    每个维度的取值区间列表，例如
                   [(-1,1), (-1,1)]，长度应与 dim 相同；
                   如果为 None，则各维默认 (-1,1)
        device:    返回的坐标 tensor 所在设备

    返回：
        coords: shape (N, dim)，其中 N = 所有 n_coords 之积
    """
    if not isinstance(n_coords, (list, tuple)):
        n_coords = (n_coords,)

    if len(n_coords) != dim:
        raise ValueError("len(n_coords) 必须等于 dim")

    if ranges is None:
        ranges = [(-1.0, 1.0)] * dim
    if len(ranges) != dim:
        raise ValueError("len(ranges) 必须等于 dim")

    seqs = []
    for i, n in enumerate(n_coords):
        v0, v1 = ranges[i]
        # 中心对齐的均匀网格
        r = (v1 - v0) / (2.0 * n)
        seq = v0 + r + (2.0 * r) * torch.arange(n, device=device)
        seqs.append(seq)

    # 构建网格
    meshes = torch.meshgrid(*seqs, indexing="ij")
    grid = torch.stack(meshes, dim=-1)  # (..., dim)
    coords = grid.reshape(-1, dim)      # (N, dim)
    return coords
