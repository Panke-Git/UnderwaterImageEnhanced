
import math
import torch
import torch.nn as nn
import functools
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


class PreProcess(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(PreProcess, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.mlp = NRNHead().cuda()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        inter = self.model(input)
        fin = self.mlp(inter)
        return fin


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

