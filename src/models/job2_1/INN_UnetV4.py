"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: INN_UnetV4.py
    @Time: 2025/11/29 23:45
    @Email: None
"""
import torch
import torch.nn as nn
from .block.ColorStructureEncoder import ColorStructureEncoder
from .block.SIREN import SirenNet
from .block.ECANet import ECALayer


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
    coords = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    coords = coords.view(1, h * w, 2)  # (1,N,2)
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


class PixelMLP(nn.Module):
    """
    逐像素共享参数的 MLP:
        输入:  (B,C_in,H,W)
        输出:  (B,C_out,H,W)
    """

    def __init__(self, in_dim, hidden_dims=(256, 256), out_dim=3):
        super(PixelMLP, self).__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (BHW,C)
        x = self.net(x)  # (BHW,out_dim)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B,out,H,W)
        return x


class SirenColorECAOnlyBlock(nn.Module):
    """
    你描述的版本：
        - 两个分支：ColorStructureEncoder + SIREN
        - 先 concat 后过 ECA 得到 B
        - 再将 B 和 SIREN 输出 concat，送入 MLP 输出最终图像
        - 没有第三个 NRN 位置编码分支

    输入:
        x: (B,3,H,W)  如 (B,3,256,256)

    输出:
        out:      (B,out_channels,H,W)  默认 out_channels=3
        B_feat:   (B,Cb,H,W)  即 ECA 的输出 B
        feat_siren: (B,Cs,H,W) SIREN 分支特征（方便你调试/可视化）
    """

    def __init__(self,
                 siren_hidden=64,
                 siren_layers=3,
                 color_channels=9,  # ColorStructureEncoder 输出通道数
                 eca_k_size=3,
                 mlp_hidden=(256, 256),
                 out_channels=3):
        super(SirenColorECAOnlyBlock, self).__init__()

        # 假设你已经实现好了这个类：输出 (B,color_channels,H,W)
        self.color_enc = ColorStructureEncoder(use_wavepool=True)

        # SIREN：输入 5 维 (RGB 3 + coord 2)，输出 siren_hidden 通道
        # self.siren = SirenNet(
        #     in_features=5,
        #     hidden_features=siren_hidden,
        #     hidden_layers=siren_layers,
        #     out_features=siren_hidden,
        #     w0=1.0,
        #     w0_initial=30.0,
        #     c=6.0,
        #     outermost_linear=True,
        #     final_activation=None,
        #     bias=True
        # )

        # ECA 作用在 [SIREN, Color] concat 后的通道上
        self.total_fuse_channels = color_channels
        self.eca = ECALayer(self.total_fuse_channels, k_size=eca_k_size)

        # ECA 输出 B_feat 再与 SIREN 特征 concat 之后的通道数：
        # C_concat2 = C_B + C_siren = (siren_hidden + color_channels) + siren_hidden
        # self.mlp_in_channels = self.total_fuse_channels + siren_hidden
        self.mlp_in_channels = self.total_fuse_channels
        self.conv_color = nn.Sequential(
            nn.Conv2d(self.mlp_in_channels, self.mlp_in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mlp_in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mlp_in_channels // 2, self.mlp_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mlp_in_channels),
            nn.ReLU(inplace=True),
        )

        # 逐像素 MLP：对 [B, feat_siren] 做映射到 out_channels
        self.pixel_mlp = PixelMLP(
            in_dim=self.mlp_in_channels,
            hidden_dims=mlp_hidden,
            out_dim=out_channels
        )

    def forward(self, x):
        """
        x: (B,3,H,W)
        """
        B, C, H, W = x.shape
        device = x.device

        # ---------- 1) 色彩结构分支 ----------
        feat_color = self.color_enc(x)  # (B,color_channels,H,W)
        _, Cc, Hc, Wc = feat_color.shape
        assert Hc == H and Wc == W, "ColorStructureEncoder 输出尺寸应与输入一致"

        # ---------- 2) SIREN 分支 ----------
        N = H * W

        # (B,3,H,W) -> (B,N,3)
        rgb_flat = x.permute(0, 2, 3, 1).reshape(B, N, 3)

        # # 坐标: (1,N,2) -> (B,N,2)
        # coord = create_coord_grid(H, W, device=device, range_type='-1_1')   # (1,N,2)
        # coord = coord.expand(B, N, 2)                                          # (B,N,2)

        # SIREN 输入 [RGB, x, y]
        # siren_in = torch.cat([rgb_flat, coord], dim=-1)                        # (B,N,5)
        # siren_out = self.siren(siren_in)                                       # (B,N,Cs)
        # Cs = siren_out.shape[-1]

        # reshape 回 feature map
        # feat_siren = siren_out.view(B, H, W, Cs).permute(0, 3, 1, 2)           # (B,Cs,H,W)

        # ---------- 3) 第一次 concat + ECA 得到 B ----------
        # fuse1 = torch.cat([feat_siren, feat_color], dim=1)                     # (B,Cs+Cc,H,W)
        fuse1 = feat_color
        fuse1 = self.conv_color(fuse1)

        B_feat = self.eca(fuse1)  # (B,Cs+Cc,H,W)

        # ---------- 4) 第二次 concat: [B, SIREN] → MLP ----------
        # fuse2 = torch.cat([B_feat, feat_siren], dim=1)                         # (B,Cs+Cc+Cs,H,W)
        out = self.pixel_mlp(B_feat)  # (B,out_channels,H,W)

        return out


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


class INN_UNetV4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64):
        super(INN_UNetV4, self).__init__()
        self.model_name = 'INN_UNetV4'

        self.per_model = SirenColorECAOnlyBlock(siren_hidden=64, siren_layers=3, color_channels=9, eca_k_size=3,
                                                mlp_hidden=(256, 256), out_channels=3)

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
        out = self.per_model(x)
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
