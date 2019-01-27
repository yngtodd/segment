import torch

from torch import nn
import torch.nn.functional as F


class DoubleBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
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
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        x, indices = self.pool(x)
        return x, indices


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet2D, self).__init__()
        self.inconv = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2, indices1 = self.down1(x1)
        x3, indices2 = self.down2(x2)
        x4, indices3 = self.down3(x3)
        x5, indices4 = self.down4(x4)
        x = self.up1(x5, indices4) 
        x = self.up2(x, indices3)
        x = self.up3(x, indices2)
        x = self.up4(x, indices1)
        x = self.outconv(x)
        x = torch.sigmoid(x)
        return x
