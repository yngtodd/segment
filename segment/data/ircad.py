import os
import pydicom
import numpy as np

from natsort import natsorted
from torch.utils.data import Dataset


class Patient:
    """
    3D-IRCADb-01 patient.
    
    Parameters
    ----------
    path : str
        Path to patient records.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path):
        self.path = path
        self.dicoms = self._list_dicoms()

    def __repr__(self):
        metadata = pydicom.read_file(self.dicoms[0])
        message = f'Patient: {metadata.PatientName}, '\
                  f'DOB: {metadata.PatientBirthDate}, '\
                  f'Sex: {metadata.PatientSex}, '\
                  f'Space between slices: {metadata.SpacingBetweenSlices:.5f}'
        return message

    def _list_dicoms(self):
        """Get dicom paths in proper order"""
        dicompath = os.path.join(self.path, 'PATIENT_DICOM')
        dicoms = [os.path.join(dicompath, img) for img in os.listdir(dicompath)] 
        # os sorts things lexicographically
        dicoms = natsorted(dicoms)
        return dicoms 

    def load_3d(self):
        """"""
        imgs = [pydicom.read_file(dicom) for dicom in self.dicoms]
        data = np.stack([img.pixel_array for img in imgs])
        return data
        

class IRCAD(Dataset):
    """
    3D-IRCADb-01 dataset.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self):
        pass

