import numpy as np
from scipy.optimize import curve_fit
from .models import unimodal, bimodal
import matplotlib.pyplot as plt
from utilities.HARD_CODED_VARIABLES import UNIMODAL_MEAN, HIST_BIN_NUMBER, UNIMODAL_AMPLITUDE, BIMODAL_AMPLITUDE_1, BIOMODAL_AMPLITUDE_2


def model_fitting(data: list, std_dev, alpha):
    
    x_data = np.linspace(min(data), max(data), num=len(data))
    y_data = np.array(data)

    n, bins, _ = plt.hist(y_data, bins=HIST_BIN_NUMBER, density=True)
    y_hist = (bins[:-1] + bins[1:]) / 2


    params_uni, _ = curve_fit(lambda x, amplitude: 
                              unimodal(x, UNIMODAL_MEAN, std_dev, amplitude), 
                              y_hist, n, p0=[UNIMODAL_AMPLITUDE])
    
    params_bi, _ = curve_fit(lambda x, amplitude1, amplitude2: 
                             bimodal(x, alpha, std_dev, amplitude1, amplitude2), 
                             y_hist, n, p0=[BIMODAL_AMPLITUDE_1, BIOMODAL_AMPLITUDE_2])

    return y_hist, n, params_uni, params_bi
