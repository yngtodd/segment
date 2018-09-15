import os
import warnings

import torch
import pydicom
import numpy as np

from natsort import natsorted
from torch.utils.data import Dataset


class ZeroMasksError(Exception):
    pass


class Patient:
    """
    3D-IRCADb-01 patient.

    Parameters
    ----------
    path : str
        Path to patient records.

    tissue : str, optional
        Type of tissue mask to load.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path, tissue=None, binarymask=False):
        self.path = path
        self.tissue = tissue
        self.binarymask = binarymask
        self.dicoms = self._list_dicoms()
        self.masks = self._list_masks()

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

    def _list_masks(self):
        """
        Get mask paths for a particular tissue.

        If no tissue is specified, these paths will not be set.

        Returns
        -------
        masks : list
            List of mask paths for a tissue.
        """
        if self.tissue:
            maskpath = os.path.join(self.path, 'MASKS_DICOM')
            maskpath = os.path.join(maskpath, self.tissue)
        else:
            maskpath = os.path.join(self.path, 'LABELLED_DICOM')

        masks = [os.path.join(maskpath, img) for img in os.listdir(maskpath)]
        # os sorts things lexicographically
        masks = natsorted(masks)
        return masks

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

    def load_masks(self):
        """
        Load all masks for a patient.

        Returns
        -------
        masks : list of np.arrays or torch.tensors
             All 2D segmentation masks of a given tissue for a patient.
             Note: If using the original masks, they remain in np.arrays.
             If binary masks are used, they are returned as torch.tensors.
        """
        masks = [pydicom.read_file(mask) for mask in self.masks]
        masks = [mask.pixel_array for mask in masks]

        if self.binarymask:
            # Not all masks in IRCAD are binary
            if not self.tissue:
                raise ValueError(f'Binary masks are not supported for multiple masks!')

            masks = [torch.tensor(mask) for mask in masks]
            ones = torch.ones_like(masks[0])
            masks = [torch.where(mask > 0, ones, mask) for mask in masks]

        return masks


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

    Instance of all patients' 2D slices of dicom images. Labels are masks for
    either a single tissue, or, if no tissue is specified, for all tissues.

    Parameters
    ----------
    path : str
        Path to IRCAD dataset.

    tissue : str, optional
        Type of tissue to segment. Options found in IRCAD `/MASKS_DICOM`.

    transform : Pytorch transforms, optional
        Pytorch transfrom for images.

    References
    ----------
    https://www.ircad.fr/research/3d-ircadb-01/
    """
    def __init__(self, path, tissue=None, binarymask=False, transform=None):
        self.ircad = IRCAD(path)
        self.tissue = tissue
        self.binarymask = binarymask
        self.transform = transform
        self.slices = self._load_slices()
        self.masks = self._load_masks()

        if self.tissue and len(self.masks) == 0:
            raise ZeroMasksError(f'There are no patients that have masks for the tissue: {self.tissue}!')

        if len(self.masks) < 50:
            warnings.warn(f'There are only {len(self.masks)} masks for {self.tissue}') 

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

    def _load_masks(self):
        """
        Loads all 2D segmentation masks in memory.

        If a given tissue is specified, then only that tissue mask will
        be loaded. If no tissue is specified, then all masks will be loaded.

        Returns
        -------
        all_labels : list of np.ndarrays
            All 2D segmentation masks of a given tissue for each patient.
        """
        all_masks = []
        for path in self.ircad.patients:
            try:
                patient = Patient(path, self.tissue, self.binarymask)
                all_masks.extend(patient.load_masks())
            except:
                FileNotFoundError('Patient {path} does not have masks for {self.tissue}')
                pass 

        return all_masks

    def __getitem__(self, idx):
        img = self.slices[idx]
        label = self.masks[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
