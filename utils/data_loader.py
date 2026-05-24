import numpy as np


def load_data(path):
    return np.load(path)


def binarize(R, threshold=1):
    return (R >= threshold).astype(int)
