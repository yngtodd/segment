import torch
from torch.nn.modules.loss import _Loss

from . import functional as F


class BinaryDiceLoss(_Loss):
    """
    Binary Dice Loss.
    """
    def __init__(self, weight=None, size_average=None, 
                 reduce=None, reduction='elementwise_mean'):
        super(BinaryDiceLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight

    def forward(self, input, target):
        return F.binary_dice_loss(input, target, weight=self.weight)
