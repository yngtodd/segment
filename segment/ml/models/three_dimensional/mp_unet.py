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
        self.pool = nn.MaxPool3d(2, return_indices=True)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        x, indices = self.pool(x)
        x = self.block(x)
        return x, indices


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.unpool = nn.MaxUnpool3d(2)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
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
        xshape = x.shape
        print(f'xshape: {xshape}')
        x1, indices1 = self.down1(x)
        x1shape = x1.shape
        print(f'x1shape: {x1shape}')

        # First transfer
        x1 = x1.to('cuda:1')
        x2, indices2 = self.down2(x1)
        x2shape = x2.shape
        print(f'x2shape: {x2shape}')
        x3, indices3 = self.down3(x2)
        x3shape = x3.shape
        print(f'x3shape: {x3shape}')

        # Second transfer
        x3 = x3.to('cuda:2')
        x4, indices4 = self.down4(x3)
        x4shape = x4.shape
        print(f'x4shape: {x4shape}')

        # Third transfer
        x4, indices4 = x4.to('cuda:3'), indices4.to('cuda:3')
        x5 = self.up1(x4, indices4, x4shape)
        print('pass')

        # Fourth transfer
        x5 = x4.to('cuda:1')
        x5 = self.up2(x5, indices3, x3shape)
        x5 = self.up3(x5, indices2, x2shape)

        # Fifth transfer
        x6 = x5.to('cuda:0')
        x6 = self.up4(x6, indices1, x1shape)
        x6 = self.outconv(x6)
        x6 = torch.sigmoid(x6)
        return x6
