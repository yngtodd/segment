import torch
from torch import nn
import torch.nn.functional as F

from segment.api import Model


class ResidualBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return F.relu(x)


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class ResidualUNet2D(Model):

    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()
        self.inconv = InConv(n_channels, 128)
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 128)
        self.down4 = Down(128, 128)
        self.up1 = Up(128, 128)
        self.up2 = Up(256, 256)
        self.up3 = Up(384, 384)
        self.up4 = Up(512, 512)
        self.outconv = OutConv(512, n_classes)

    def concat_channels(self, x_cur, x_prev):
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        x = self.inconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x)
        x = self.concat_channels(x, x4)
        x = self.up2(x)
        x = self.concat_channels(x, x3)
        x = self.up3(x)
        x = self.concat_channels(x, x2)
        x = self.up4(x)
        x = self.outconv(x)
        x = torch.sigmoid(x)
        return x
