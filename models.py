""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def get_position_embeddings(t, device):
    x = positionalencoding1d(512, 1000).to(device)
    emb = x[t]
    return emb


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.mlp1 = nn.Linear(784 + 512, 512)
        self.b1 = nn.BatchNorm1d(512)
        self.mlp2 = nn.Linear(512 + 512, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.mlp3 = nn.Linear(256 + 512, 256)
        self.b3 = nn.BatchNorm1d(256)
        self.mlp4 = nn.Linear(256 + 512, 784)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = torch.flatten(x, 1, 3)  # B*C*H*W -> B*(C*H*W)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp3(x)
        x = self.b3(x)
        x = self.relu(x)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp4(x)
        x = x.view(-1, 1, 28, 28)
        return x


class SimpleMLP2(nn.Module):
    def __init__(self):
        super(SimpleMLP2, self).__init__()
        self.mlp1 = nn.Linear(2 + 64, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.mlp2 = nn.Linear(32 + 64, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.mlp3 = nn.Linear(8 + 64, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, t):
        x = torch.flatten(x, 1, 3)  # B*C*H*W -> B*(C*H*W)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = torch.cat((x, t), dim=-1)
        x = self.mlp3(x)
        x = self.tanh(x)
        x = x.view(-1, 2, 1, 1)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        input_size = [512, 64, 128, 256, 512, 1024, 512, 256, 128, 64]
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_size[0], input_size[i + 1])
                for i in range(len(input_size) - 1)
            ]
        )

    def forward(self, x, t):
        x1 = self.inc(x)
        t1 = self.linears[0](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + t1
        x2 = self.down1(x1)
        t1 = self.linears[1](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x2 = x2 + t1
        x3 = self.down2(x2)
        t1 = self.linears[2](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x3 = x3 + t1
        x4 = self.down3(x3)
        t1 = self.linears[3](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x4 = x4 + t1
        x5 = self.down4(x4)
        t1 = self.linears[4](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x5 = x5 + t1
        x = self.up1(x5, x4)
        t1 = self.linears[5](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x = x + t1
        x = self.up2(x, x3)
        t1 = self.linears[6](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x = x + t1
        x = self.up3(x, x2)
        t1 = self.linears[7](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x = x + t1
        x = self.up4(x, x1)
        t1 = self.linears[8](t)
        t1 = t1.unsqueeze(-1).unsqueeze(-1)
        x = x + t1
        logits = self.outc(x)
        return logits
