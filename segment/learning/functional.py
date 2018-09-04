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
