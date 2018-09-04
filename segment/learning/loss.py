import torch


def dice_loss(output, target):
    """
    Dice loss.

    Parameters
    ----------
    output : torch tensor
        Output of model.

    target : torch tensor
        True mask of the image.

    References
    ----------
    https://github.com/pytorch/pytorch/issues/1249
    """
    output = torch.sigmoid(output)
    smooth = 1.

    output = output.view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()
    
    return 1 - ((2. * intersection + smooth) /
           (output.sum() + target.sum() + smooth))
