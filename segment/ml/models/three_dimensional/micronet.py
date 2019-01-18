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
        self.pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x):
        shape = x.shape
        x = self.block(x)
        x, indices = self.pool(x)
        return x, indices, shape


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.unpool = nn.MaxUnpool3d(3, stride=2)
        self.block = DoubleBlock(in_ch, out_ch)

    def forward(self, x, indices, output_shape):
        x = self.block(x)
        x = self.unpool(x, indices, output_shape)
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
        self.inconv = InConv(n_channels, 2)
        self.down1 = Down(2, 4)
        self.up1 = Up(4, 2)
        self.outconv = OutConv(2, n_classes)

    def forward(self, x):
        x = self.inconv(x)
        #xshape = x.shape
        #print(f'xshape: {xshape}')
        x1, indices1, x1shape = self.down1(x)
        #x1shape = x1.shape
        print(f'x1shape: {x1shape}')
        print(f'indices1: {indices1.shape}')

        # Third transfer
        #x4, indices4 = x4.to('cuda:3'), indices4.to('cuda:3')
        x2 = self.up1(x1, indices1, output_shape=x1shape)
        print('pass')

        x3 = self.outconv(x2)
        x3 = torch.sigmoid(x3)
        return x6
