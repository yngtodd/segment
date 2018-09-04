import os
import torch


def save_checkpoint(path, state):
    """
    Save Pytorch model and optimizer state.

    Parameters
    ----------
    path : str
        Path to save state to.

    state : dict
        Dictionary of save state. Must include `epoch` number
        and, ideally, both the model state dict and optimizer state dict.
    """
    filename='checkpoint' + str(state['epoch']) + '.pth.tar'
    savepath = os.path.join(path, filename)
    torch.save(state, savepath)
