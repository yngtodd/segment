import torch
import torch.nn as nn
import torch.nn.functional as _F
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


class MulticlassDiceLoss(nn.Module):
	"""
	requires one hot encoded target. Applies DiceLoss on each class iteratively.
	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
	  batch size and C is number of classes
	"""
	def __init__(self):
		super(MulticlassDiceLoss, self).__init__()
 
	def forward(self, input, target, weights=None):
 
		C = target.shape[1]
 
		dice = SoftDiceLoss()
		totalLoss = 0
 
		for i in range(C):
			diceLoss = dice(input[:,i], target[:,i])
			if weights is not None:
				diceLoss *= weights[i]
			totalLoss += diceLoss
 
		return totalLoss


class SoftDiceLoss(nn.Module):
	def __init__(self):
		super(SoftDiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
        denom_sum = input_flat.sum(1) + target_flat.sum(1)
        
        if denom_sum == 0:
            loss = 0
        else:
            loss = 2 * (intersection.sum(1) + smooth) / denom_sum
            loss = 1 - loss.sum() / N
        
        return loss