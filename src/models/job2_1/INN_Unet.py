"""
    @Project: UnderwaterImageEnhanced
    @Author: paxton
    @FileName： INN_Unet.py
    @Date：2025/11/27 22:05
    @OS：
    @Email: None
"""
import math
import torch
import torch.nn as nn
from .block import ColorStructureEncoder
from .block.SIREN import SirenNet
from .block.ECANet import ECALayer

# 与原 NRN 一致的设置
hidden_list = [256, 256, 256]
L = 8  # 位置编码的频率数


# ===============================
# 2. SIREN + Color + Cross Block
# ===============================

# 构建坐标
def create_coord_grid(h, w, device, range_type='-1_1'):
    """
    创建 2D 坐标网格，用于给 SIREN 输入坐标：
    返回：(1, H*W, 2)，便于 expand 到 B
    """
    if range_type == '-1_1':
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
    else:
        ys = torch.linspace(0.0, 1.0, h, device=device)
        xs = torch.linspace(0.0, 1.0, w, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    coords = torch.stack([xx, yy], dim=-1)          # (H,W,2)
    coords = coords.view(1, h * w, 2)               # (1,N,2)
    return coords


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

class SirenColorNRNBlock(nn.Module):
    """
    SIREN + ColorStructureEncoder + ECA 通道注意力 + NRNHead

    输入:
        x: (B,3,H,W) 例如 (B,3,256,256)

    流程:
        1) ColorStructureEncoder(x) -> feat_color: (B,Cc,H,W)
        2) SIREN分支 -> feat_siren: (B,Cs,H,W)
        3) 融合: concat(feat_siren, feat_color) -> ECA -> 1x1 conv -> B_feat
        4) NRNHead(B_feat) 做 X->X' + concat + MLP -> out: (B,out_channels,H,W)
    """

    def __init__(self,
                 siren_hidden=64,
                 siren_layers=3,
                 color_channels=9,   # ColorStructureEncoder 输出通道
                 cross_dim=64,       # 融合后希望的通道数，给 NRNHead 用
                 out_channels=3):
        super(SirenColorNRNBlock, self).__init__()

        # 1) 色彩结构编码器（已改成输出 (B,9,H,W)）
        self.color_enc = ColorStructureEncoder(use_wavepool=True)

        # 2) SIREN: 输入 5 维 (RGB 3 + coord 2)，输出 siren_hidden 通道
        self.siren = SirenNet(
            in_features=5,
            hidden_features=siren_hidden,
            hidden_layers=siren_layers,
            out_features=siren_hidden,
            w0=1.0,
            w0_initial=30.0,
            c=6.0,
            outermost_linear=True,
            final_activation=None,
            bias=True
        )

        # 3) ECA 通道注意力，用在 concat 后的特征上
        self.total_channels = siren_hidden + color_channels
        self.eca = ECALayer(self.total_channels, k_size=3)

        # 把 concat+ECA 后的通道数映射到 cross_dim，方便后面 NRNHead 使用
        self.fuse_conv = nn.Conv2d(self.total_channels, cross_dim, kernel_size=1, bias=True)

        # 4) NRNHead: feat + X' -> MLP -> out
        self.nrn_head = NRNHead(
            feat_channels=cross_dim,
            out_channels=out_channels,
            L_freq=L,
            hidden=hidden_list
        )

    def forward(self, x):
        """
        x: (B,3,H,W)
        返回:
            out:    (B,out_channels,H,W)
            B_feat: (B,cross_dim,H,W)
        """
        B, C, H, W = x.shape
        device = x.device

        # -------- 1) ColorStructureEncoder 分支 --------
        feat_color = self.color_enc(x)              # (B,color_channels,H,W)
        _, Cc, Hc, Wc = feat_color.shape
        assert Hc == H and Wc == W, "ColorStructureEncoder 输出分辨率应与输入一致"

        # -------- 2) SIREN 分支 --------
        N = H * W

        # (B,3,H,W) -> (B,N,3)
        rgb_flat = x.permute(0, 2, 3, 1).reshape(B, N, 3)

        # 坐标: (1,N,2) -> (B,N,2)
        coord = create_coord_grid(H, W, device=device, range_type='-1_1')  # (1,N,2)
        coord = coord.expand(B, N, 2)                                         # (B,N,2)

        # SIREN 输入: [RGB, x, y] -> (B,N,5)
        siren_in = torch.cat([rgb_flat, coord], dim=-1)
        siren_out = self.siren(siren_in)                                      # (B,N,Cs)

        # reshape -> (B,Cs,H,W)
        Cs = siren_out.shape[-1]
        feat_siren = siren_out.view(B, H, W, Cs).permute(0, 3, 1, 2)          # (B,Cs,H,W)

        # -------- 3) ECA 融合（替代 Cross-Attention） --------
        # 通道拼接: (B,Cs+Cc,H,W)
        feat_fuse = torch.cat([feat_siren, feat_color], dim=1)                # (B,Cs+Cc,H,W)

        # ECA 通道注意力
        feat_eca = self.eca(feat_fuse)                                        # (B,Cs+Cc,H,W)

        # 1x1 conv 降维/映射到 cross_dim
        B_feat = self.fuse_conv(feat_eca)                                     # (B,cross_dim,H,W)

        # -------- 4) NRNHead: B_feat + X' -> MLP -> out --------
        out = self.nrn_head(B_feat)                                           # (B,out_channels,H,W)

        # 这里没有注意力矩阵了，就不返回 attn_weights
        return out, B_feat


class ConvBlock(nn.Module):
    """Conv => BN => ReLU x2"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class INN_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(INN_UNet, self).__init__()
        self.model_name = 'INN_Unet'

        self.per_model = SirenColorNRNBlock(siren_hidden=64, siren_layers=3, color_channels=9,cross_dim=64, out_channels=3)

        # Down path
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)

        # Up path
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        # Output
        self.out_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        out, B_feat = self.per_model(x)
        x = out
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        out = self.out_conv(x)
        out = torch.sigmoid(out)  # Normalize output to [0,1]
        return out




def test_siren_color_nrn_block_eca():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = INN_UNet().to(device)

    model.eval()
    x = torch.randn(1, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(x)

    print("Input x shape: ", x.shape)
    print("out shape:     ", out.shape)

    assert out.shape == (1, 3, 256, 256)
    print("✅ SirenColorNRNBlockECA 256x256 forward OK")

#
# if __name__ == "__main__":
#     test_siren_color_nrn_block_eca()