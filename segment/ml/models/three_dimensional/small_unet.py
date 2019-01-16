import torch

from torch import nn
import torch.nn.functional as F


class DoubleBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.MaxUnpool3d(2),
            DoubleBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.up(x)


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.inconv = InConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.up1 = Up(256, 32)
        self.up2 = Up(64, 16)
        self.up3 = Up(32, 8)
        self.up4 = Up(16, 8)
        self.outconv = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outconv(x)
        x = torch.sigmoid(x)
        return x
