import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, savepath, savefile):
    """
    Save Pytorch model and optimizer state.

    Parameters
    ----------
    model : nn.Module
        Model to be saved.

    optimizer : pytorch optimizer
        Optimizer to be saved.

    epoch : int
        Epoch save occurs.

    loss : float
        Current loss.

    savepath : str
        Path to save state to.

    savefile : str
        Filename to save to.
    """
    state = {
      'epoch': epoch,
      'model_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'loss': loss
    }
    savefile = os.path.join(path, savefile)
    torch.save(state, savefile)


def load_checkpoint(model, optimizer, savepath, savefile):
    """
    Load previously saved model and optimizer.

    Paramters
    ---------
    model : nn.Module
        Model to be loaded.

    optimizer : pytorch optimizer
        Optimizer to be restored.

    savepath :
        Path to directory where checkpoint file lives.

    savefile : str
        Name of the file checkpoint file.

    Returns
    -------
        Restored model and optimizer.
    """
    savefile = os.path.join(savepath, savefile)
    try:
        checkpoint = torch.load(savefile)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer
    except FileNotFoundError as fnf_error:
        print(fnf_error)