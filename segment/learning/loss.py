import torch
from torch.nn.modules.loss import _Loss

import segment.learning.functional as F


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


class SoftDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
