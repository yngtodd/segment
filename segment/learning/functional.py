import torch


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


def dice_coefficient(input, target, smooth=1):
    """
    Compute dice coefficient.
    """
    input = torch.sigmoid(input)
    batch_size = input.size(0)
    input = input.view(batch_size, -1)
    target = target.view(batch_size, -1)
    intersection = (input * target).sum()

    return ((2. * intersection + smooth) /
           (input.sum() + target.sum() + smooth))


def dice_coeff(input, target):
    """
    Alternative Dice coefficient

    References
    ----------
    https://github.com/pytorch/pytorch/issues/1249
    """
    batch_size = input.size(0)
    smooth = 1.

    pred = input.view(batch_size, -1)
    truth = target.view(batch_size, -1)

    intersection = (pred * truth).sum(1)

    dice = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return dice.mean().item()


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
