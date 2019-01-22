import math
import torch
import torch.nn.functional as F

import numpy as np
from scipy import spatial


def binary_dice_loss(input, target, smooth=1, weight=None):
    """
    Binary Dice loss.

    Parameters
    ----------
    output : torch tensor
        Output of model.

    target : torch tensor
        True mask of the image.

    smooth : float
        Smoothing factor.

    References
    ----------
    https://github.com/pytorch/pytorch/issues/1249
    """
    input = torch.sigmoid(input)

    input = input.view(-1)
    target = target.view(-1)
    intersection = (input * target).sum()

    loss = 1 - ((2. * intersection + smooth) /
           (input.sum() + target.sum() + smooth))

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss

    if size_average:
        return loss.mean()

    return loss


def dice_coefficient(input, target):
    """
    Compute Dice coefficient

    References
    ----------
    https://github.com/pytorch/pytorch/issues/1249
    """
    batch_size = input.size(0)
    smooth = 1.

    input = F.sigmoid(input)

    pred = input.view(batch_size, -1)
    truth = target.view(batch_size, -1)

    intersection = (pred * truth).sum(1)

    dice = 2. * (intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return dice.mean().item()


def dice_score(prediction, groundtruth):
    prediction = torch.sigmoid(prediction)
    pred = prediction.cpu()
    pred = pred.detach().numpy()
    pred = threshold_predictions(pred)
    groundtruth = groundtruth.cpu().numpy()
    pflat = pred.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d


def continuous_dice_coefficient(output, target):
    """
    Continuous version of the Dice coefficient.

    References
    ----------
    https://www.biorxiv.org/content/biorxiv/early/2018/04/25/306977.full.pdf
    """
    pred = torch.sigmoid(output)

    batch_size = output.size(0)
    pred = pred.view(batch_size, -1)
    truth = target.view(batch_size, -1)

    pred_size = pred.sum(1)
    truth_size = truth.sum(1)

    intersection = (pred * truth).sum(1)

    if intersection > 0:
        c =  (intersection / (truth * (1-math.exp(1e3*pred))).sum()).sum()
    else:
        c = 1

    continuous_dice = (2 * intersection) / (c * truth_size + pred_size)

    return continuous_dice


def jaccard_index(input, target):
    """
    Compute Jaccard index.

    References
    ----------
    https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
    """
    intersection = (input*target).long().sum().data.cpu()[0]
    union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection

    if union == 0:
        return float('nan')
    else:
        return float(intersection) / float(max(union, 1))


def iou_pytorch(output: torch.Tensor, target: torch.Tensor):
    """
    Compute intersection of unions.

    Parameters
    ----------
    output : torch.Tensor
        Output from model

    target : torch.Tensor
        True mask.
    """
    smooth = 1e-6
    outputs = outputs.squeeze(1)

    intersection = (output & target).sum((1, 2)).float()
    union = (output | target).sum((1, 2)).float()

    iou = (intersection + smooth) / (union + smooth)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Parameters
    ----------
        true: a tensor of shape [B, 1, H, W].

        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.

        eps: added to the denominator for numerical stability.

    Returns
    -------
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)

    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice), dice
