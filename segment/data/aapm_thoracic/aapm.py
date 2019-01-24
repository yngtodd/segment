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
        self.patient_number = patient_number
        self.img = f'image_{patient_number}.pth'
        self.mask = f'label_{patient_number}.pth'

    def __repr__(self):
        img = self.load_img()
        return f'AAPM Thoracic Patient: {self.patient_number} '\
               f'\n Image Size: {img.shape}'

    def load_img(self):
        img_path = os.path.join(self.path, self.img)
        return torch.load(img_path)

    def load_mask(self):
        mask_path = os.path.join(self.path, self.mask)
        try:
            return torch.load(mask_path)
        except:
            raise FileNotFoundError(f'Patient {self.patient_number} '\
                                    f'has no label mask.')

class AAPM(Dataset):
    """
    AAPM Thoracic Dataset.

    Parameters
    ----------
    path : str
        Path to AAPM data.

    References
    ----------
    """
    def __init__(self, path, tissue=None, transform=None):
        self.path = path
        self.transform = transform

    def __repr__(self):
        return f'AAPM Thoracic Dataset.'

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        patient = Patient(self.path, idx)
        img = patient.load_img()
        mask = patient.load_mask()

        if self.transform is not None:
            img = self.transform(img)

        return img, mask