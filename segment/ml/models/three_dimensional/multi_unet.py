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
        x = self.block(x)
        x, indices = self.pool(x)
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


class ModelParallelUNet3D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(MicroUNet3D, self).__init__()
        self.inconv = InConv(n_channels, 2).to('cuda:0')
        self.down1 = Down(2, 4).to('cuda:0')
        self.down2 = Down(4, 8).to('cuda:1')
        self.down3 = Down(16, 32).to('cuda:1')
        self.down4 = Down(32, 64).to('cuda:2')
        self.up1 = Up(64, 32).to('cuda:2')
        self.up2 = Up(32, 16).to('cuda:3')
        self.up3 = Up(8, 4).to('cuda:3')
        self.up4 = Up(4, 2).to('cuda:4')
        self.outconv = OutConv(2, n_classes).to('cuda:4')


    def forward(self, x):
        # GPU 0
        x1 = self.inconv(x)
        x2, indices1 = self.down1(x1)
        # Transfer x -> GPU:1. & idx -> GPU:4
        x3 = x2.to('cuda:1')
        indices1 = indices1.to('cuda:4')
        # GPU 1
        x3, indices2 = self.down2(x2)
        x4, indices3 = self.down3(x3)
        # Transfer to next GPU:2 & idx2, idx3 -> GPU:3
        x4  = x4.to('cuda:2') 
        indices2, indices3 = indices2.to('cuda:3'), indices3.to('cuda:3')
        # GPU 2
        x5, indices4 = self.down4(x4, indices3, x3.shape)
        x6 = self.up1(x5, indices4, x4.shape)
        # Transfer x -> GPU:3.
        x6 = x6.to('cuda:3')
        # GPU 3
        x7 = self.up2(x6, indices3, x3.shape)
        x8 = self.up3(x7, indices2, x2.shape)
        # Transfer x -> GPU:4
        x8 = x8.to('cuda:4')
        # GPU 4 
        x9 = self.up4(x8, indices1, x1.shape)
        x10 = self.outconv(x9)
        return x10