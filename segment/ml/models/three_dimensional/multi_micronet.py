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


class MicroUnet3D(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(MicroUnet3D, self).__init__()
        self.inconv = InConv(n_channels, 2).to('cuda:0')
        self.down1 = Down(2, 4).to('cuda:0')
        self.down2 = Down(4, 8).to('cuda:0')
        self.up1 = Up(8, 4).to('cuda:1')
        self.up2 = Up(4, 2).to('cuda:1')
        self.outconv = OutConv(2, n_classes).to('cuda:1')


    def forward(self, x):
        x1 = self.inconv(x)
        x2, indices1 = self.down1(x1)
        x3, indices2 = self.down2(x2)

        # Transfer to next GPU.
        x2, indices1= x2.to('cuda:1'), indices1.to('cuda:1')
        x3, indices2 x3.to('cuda:1'), indices2.to('cuda:1')
        
        x4 = self.up1(x3, indices2, x2.shape)
        x5 = self.up2(x4, indices1, x1.shape)
        x6 = self.outconv(x5)
        return x6
