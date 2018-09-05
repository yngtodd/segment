import torch
import torch.nn as nn
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


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        return bce_loss + (1 - dice_coef)
