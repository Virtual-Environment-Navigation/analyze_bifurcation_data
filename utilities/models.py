import numpy as np


def unimodal(x, mean, std_dev, amplitude):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * std_dev ** 2))


def bimodal(x, mean, std_dev, amplitude1, amplitude2):
    return unimodal(x, mean, std_dev, amplitude1) + unimodal(x, -mean, std_dev, amplitude2)
