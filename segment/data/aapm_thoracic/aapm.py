import os
import numpy as np

import torch
from torch.utils.data import Dataset


class Patient:
    """
    AAPM Thoracic patient.

    Parameters
    ----------
    path : str
        Path to patient records.

    patient_number : int 
        Integer value corresponding to patient ID.
        - Must be between [0, 59]

    References
    ----------
    """
    def __init__(self, path, patient_number):
        self.path = path
        self.img = f'image_{patient_number}.pth'
        self.mask = f'label_{patient_number}.pth'

    def __repr__(self):
        img = self.load_img()
        return f'AAPM Thoracic Patient: {self.img} '\
               f'\tImage Size: {img.shape}'

    def load_img(self):
        img_path = os.path.join(self.path, self.img)
        return torch.load(img_path)

    def load_mask(self):
        mask_path = os.path.join(self.path, self.mask)
        return torch.load(mask_path)