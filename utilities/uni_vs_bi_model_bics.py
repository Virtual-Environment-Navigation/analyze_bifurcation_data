from utilities.models import unimodal, bimodal
import numpy as np
from utilities.model_fitting import model_fitting

def uni_vs_bi_model_bics(data: list, mean: float, std_dev: float, alpha: float):

    [x_data, y_data, params_uni, params_bi] = model_fitting(data, mean, std_dev, alpha)

    residuals_uni = y_data - unimodal(x_data, 0, std_dev, *params_uni)
    ss_res_uni = np.sum(residuals_uni**2)
    bic_uni = len(y_data) * np.log(ss_res_uni / len(y_data)) + len(params_uni) * np.log(len(y_data))

    print("Size of params_uni: ", len(params_uni))
    print("Size of params_bi: ", len(params_bi))

    residuals_bi = y_data - bimodal(x_data, mean, std_dev, params_bi[0], params_bi[1])
    ss_res_bi = np.sum(residuals_bi**2)
    bic_bi = len(y_data) * np.log(ss_res_bi / len(y_data)) + len(params_bi) * np.log(len(y_data))

    return bic_uni, bic_bi