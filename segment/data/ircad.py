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
        """
        Get dicom paths in proper order.
        
        Returns
        -------
        dicoms : list
            List of dicom paths for the patient.
        """
        dicompath = os.path.join(self.path, 'PATIENT_DICOM')
        dicoms = [os.path.join(dicompath, img) for img in os.listdir(dicompath)] 
        # os sorts things lexicographically
        dicoms = natsorted(dicoms)
        return dicoms 

    def load_3d(self):
        """
        Load 3D pixel array for the patient.
        
        Returns
        -------
        arry : np.ndarray 
            3D pixel array for patient's CT scan.
        """
        imgs = [pydicom.read_file(dicom) for dicom in self.dicoms]
        arry = np.stack([img.pixel_array for img in imgs])
        return arry 
        

class IRCAD(Dataset):
    """
    3D-IRCADb-01 dataset.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self):
        pass

