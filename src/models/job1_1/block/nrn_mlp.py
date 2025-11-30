
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与原 NRN 一致的设置
hidden_list = [256, 256, 256]
L = 8  # 位置编码的频率数


def make_coord(shape, ranges=None, flatten=True):
    """
    与原 NRN 完全一致的坐标生成方式：采样在 cell 中心，范围默认 [-1, 1]
    shape: (H, W)
    返回:
        flatten=True  -> (H*W, 2)
        flatten=False -> (H, W, 2)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)  # (H,W,2)

    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # (H*W,2)

    return ret


class MLP(nn.Module):
    """
    你 mlp.py 里的那个 MLP（完全照搬）
    """
    def __init__(self, in_dim, out_dim, hidden_list):
        super(MLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: (..., in_dim)
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        x = x.view(*shape, -1)
        return x


class NRNHead(nn.Module):
    """
    只保留 NRN 中你要的部分：
      - 坐标 X → 位置编码 X'
      - 与特征 B concat
      - MLP 输出 (比如 RGB)

    输入:
        feat: (B, C_feat, H, W)   # 这里的 feat 就是 Cross Attention 后的 B

    输出:
        out:  (B, C_out, H, W)    # 默认 C_out=3
    """

    def __init__(self, feat_channels, out_channels=3, L_freq=L, hidden=hidden_list):
        super(NRNHead, self).__init__()
        self.L = L_freq
        self.feat_channels = feat_channels

        # 每个坐标有 2 维 (x,y)，位置编码为 2*N*L = 4L
        in_dim = feat_channels + 2 + 4 * self.L
        self.imnet = MLP(in_dim, out_channels, hidden)

    def positional_encoding(self, coord):
        """
        与原 NRN 的 positional_encoding 保持一致:
        coord: (B, N, 2)
        返回:  (B, N, 4L)
        """
        shape = coord.shape
        device = coord.device
        freq = 2 ** torch.arange(self.L, dtype=torch.float32, device=device) * math.pi  # (L,)
        spectrum = coord[..., None] * freq            # (B,N,2,L)
        sin, cos = spectrum.sin(), spectrum.cos()     # (B,N,2,L)

        # stack 后 reshape 成 (B,N,4L)
        enc = torch.stack([sin, cos], dim=-2)         # (B,N,2,2,L)
        enc = enc.view(*shape[:-1], -1)               # (B,N,4L)
        return enc

    def forward(self, feat):
        """
        feat: (B, C_feat, H, W)
        """
        B, C, H, W = feat.shape
        assert C == self.feat_channels, "feat 通道数与 NRNHead 预期不一致"

        device = feat.device
        # 1) 生成坐标网格: (H*W,2) -> (B,H*W,2)
        coord = make_coord((H, W), ranges=None, flatten=True).to(device)  # (H*W,2)
        coord = coord.unsqueeze(0).expand(B, -1, -1)                      # (B,N,2)

        # 2) 位置编码 X'（和 NRN 一样）
        coord_enc = self.positional_encoding(coord)                       # (B,N,4L)
        coord_full = torch.cat([coord, coord_enc], dim=-1)                # (B,N,2+4L)

        # 3) 展平 B_feat: (B,C,H,W) -> (B,H*W,C)
        feat_flat = feat.permute(0, 2, 3, 1).view(B, H * W, C)            # (B,N,C_feat)

        # 4) concat(B, X') → 输入 MLP
        inp = torch.cat([feat_flat, coord_full], dim=-1)                  # (B,N,C_feat+2+4L)

        out = self.imnet(inp)                                             # (B,N,out_channels)
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)                   # (B,out_channels,H,W)
        return out
