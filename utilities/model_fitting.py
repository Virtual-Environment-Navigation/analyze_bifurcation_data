
import numpy as np
from scipy.optimize import curve_fit
from .models import unimodal, bimodal
import matplotlib.pyplot as plt


import numpy as np
from scipy.optimize import curve_fit
from .models import unimodal, bimodal
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

def model_fitting(data: list, mean, std_dev, alpha):

    mean = 0 # reset input to == 0, if input is given at all. Reason for this is that 0 is a good prediction for that conditions heading
    
    x_data = np.linspace(min(data), max(data), num=len(data))
    y_data = np.array(data)

    # BEGIN: histogram
    n, bins, patches = plt.hist(y_data, bins=30, density=True)
    x_hist = bins
    y_hist = (bins[:-1] + bins[1:]) / 2

    # Smooth out the curve
    spl = make_interp_spline(y_hist, n)
    y_smooth = spl(y_hist)

    # Plot the smoothed curve on top of the histogram
    plt.plot(y_hist, y_smooth, 'r')
    plt.plot(y_hist, n)
    plt.show()
    # END: histogram

    params_uni, _ = curve_fit(lambda x, amplitude: unimodal(x, mean, std_dev, amplitude), y_data, x_data, p0=[0.1])
    
    params_bi, _ = curve_fit(lambda x, amplitude1, amplitude2: bimodal(x, alpha, std_dev, amplitude1, amplitude2), y_data, x_data, p0=[0.05, 0.05])

    return x_data, y_data, params_uni, params_bi
