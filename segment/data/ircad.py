import os
import dicom

from torch.utils.data import Dataset


class IRCAD(Dataset):
    """
    3D-IRCADb-01 dataset.

    Parameters
    ----------
    dicoms : str
        Path to dicom images.

    masks : str
        Path to image masks.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, dicoms, masks=None):
        self.dicoms = dicoms
        self.masks = masks

