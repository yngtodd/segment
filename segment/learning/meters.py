import os
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value

    Parameters:
    ----------
    name : str
        Name of the object to be tracked.
    """
    def __init__(self, name):
        self.name = name
        self.reset()

    def __str__(self):
        return f'Average Meter for {self.name}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = np.empty(0)
        self.avgs = np.empty(0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals = np.append(self.vals, self.val)
        self.avgs = np.append(self.avgs, self.avg)

    def save(self, path):
        avgpath = os.path.join(path, self.name + '_avgs')
        valpath = os.path.join(path, self.name + '_vals')
        np.save(avgpath, self.avgs)
        np.save(valpath, self.vals)
