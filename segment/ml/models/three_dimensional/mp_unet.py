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
        self.pool = nn.MaxPool3d(2, stride=1, return_indices=True)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        x, indices = self.pool(x)
        x = self.block(x)
        return x, indices


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.unpool = nn.MaxUnpool3d(2, stride=1)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.block(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.inconv = InConv(n_channels, 2).to('cuda:0')
        self.down1 = Down(2, 4).to('cuda:0')
        self.down2 = Down(4, 8).to('cuda:1')
        self.down3 = Down(8, 16).to('cuda:1')
        self.down4 = Down(16, 32).to('cuda:2')
        self.up1 = Up(32, 16).to('cuda:3')
        self.up2 = Up(16, 8).to('cuda:1')
        self.up3 = Up(8, 4).to('cuda:1')
        self.up4 = Up(4, 2).to('cuda:0')
        self.outconv = OutConv(2, n_classes).to('cuda:0')

    def forward(self, x):
        x = self.inconv(x)
        x, indices1 = self.down1(x)

        # First transfer
        x = x.to('cuda:1')
        x, indices2 = self.down2(x)
        x, indices3 = self.down3(x)

        # Second transfer
        x = x.to('cuda:2')
        x, indices4 = self.down4(x)

        # Third transfer
        x, indices4 = x.to('cuda:3'), indices4.to('cuda:3')
        x = self.up1(x, indices4)

        # Fourth transfer
        x = x.to('cuda:1')
        x = self.up2(x, indices3)
        x = self.up3(x, indices2)

        # Fifth transfer
        x = x.to('cuda:0')
        x = self.up4(x, indices1)
        x = self.outconv(x)
        x = torch.sigmoid(x)
        return x
