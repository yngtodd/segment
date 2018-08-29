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

    def load_slices(self):
        """
        Load patient CT scan slices.

        Returns
        -------
        slices : list of np.arrays
             All 2D CT scans for a patient.
        """
        dicoms = [pydicom.read_file(dicom) for dicom in self.dicoms]
        slices = [dicom.pixel_array for dicom in dicoms]
        return slices
        

class IRCAD:
    """
    3D-IRCADb-01 dataset.

    Parameters
    ----------
    path : str
        Path to IRCAD dataset.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path):
        self.path = path
        self.patients = self._list_patients()

    def __repr__(self):
        return f'IRCAD liver segmentation dataset.' 

    def _list_patients(self):
        """
        Get patient paths in proper order.

        Returns
        -------
        patients : list
            List of patient paths for the dataset.
        """
        patients = [os.path.join(self.path, patient) for patient in os.listdir(self.path)]
        # os sorts things lexicographically
        patients = natsorted(patients)
        return patients 


class IRCAD3D(Dataset):
    """
    3D IRCAD dataset.

    Parameters
    ----------
    path : str
        Path to IRCAD dataset.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path, transform=None):
        self.ircad = IRCAD(path)
        self.transform = transform

    def __repr__(self):
        return f'IRCAD 3D liver segmentation'

    def __len__(self):
        return len(self.ircad.patients)

    def __getitem__(self, idx):
        patient_path = self.ircad.patients[idx]
        patient = Patient(patient_path)
        img = patient.load_3d()

        if self.transform is not None:
            img = self.transform(img)

        return img 


class IRCAD2D(Dataset):
    """
    2D IRCAD dataset.

    Parameters
    ----------
    path : str
        Path to IRCAD dataset.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path, transform=None):
        self.ircad = IRCAD(path)
        self.slices = self._load_slices()
        self.transform = transform

    def __repr__(self):
        return f'IRCAD 2D liver segmentation'

    def __len__(self):
        return len(self.slices)

    def _load_slices(self):
        """
        Loads all 2D CT slices in memory.

        Returns
        -------
        all_slices : list of np.ndarrays
            All 2D CT scans in natural order for each patient.
        """
        all_slices = []
        for path in self.ircad.patients:
            patient = Patient(path)
            all_slices.extend(patient.load_slices())
        return all_slices

    def __getitem__(self, idx):
        img = self.slices[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img
