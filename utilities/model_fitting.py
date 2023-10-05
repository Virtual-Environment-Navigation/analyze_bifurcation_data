
import numpy as np
from scipy.optimize import curve_fit
from .models import unimodal, bimodal

def model_fitting(data: list, mean: float, std_dev: float, alpha: float):
    
    x_data = np.linspace(min(data), max(data), num=len(data))
    y_data = np.array(data)
    
    params_uni, _ = curve_fit(lambda x, amplitude: unimodal(x, 0, std_dev, amplitude), x_data, y_data, p0=[1])
    
    params_bi, _ = curve_fit(lambda x, amplitude1, amplitude2: bimodal(x, alpha/2, std_dev, amplitude1, amplitude2), x_data, y_data, p0=[1, 1])

    return x_data, y_data, params_uni, params_bi
