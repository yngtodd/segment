import numpy as np
from math import floor

from torch.utils.data import Dataset


class DataSplitter(Dataset):
    """
    Subset dataset by index.

    Helper class to be used with `train_valid_splitter`.

    Parameters
    ----------
    data : torch.utils.data.Dataset instance

    length : int
        Number of samples in subset.

    mapping : list 
        Indices of the original data to be used in subset. 
    """
    def __init__(self, data, length, mapping):
        self.data = data 
        self.length = length
        self.mapping = mapping

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, index):
        return self.data[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(data, valpercent=.20, random_seed=None):
    """
    Split dataset into train and validation sets.

    Parameters
    ----------
    data : torch.utils.data.DataSet instance
        Dataset to be split into training and validation sets.

    valpercent : float
        Percentage of the validation set to be withheld for validation.
        Note: will take the floor of that percentage, we can't index by floats.

    random_seed : int
        Random seed for shuffling.

    Returns
    -------
    train : torch.utils.data.Dataset instance
        Training set 

    valid : torch.utils.data.Dataset instance
    """
    if random_seed!=None:
        np.random.seed(random_seed)

    datalen = len(data)
    valid_size = floor(datalen * valpercent)
    train_size = datalen - valid_size

    indices = list(range(datalen))
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]

    train = DataSplitter(data, train_size, train_mapping)
    valid = DataSplitter(data, valid_size, valid_mapping)

    return train, valid
