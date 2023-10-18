from utilities.models import unimodal, bimodal
import numpy as np
from utilities.model_fitting import model_fitting
from utilities.debug.plot_models import plot_models
from utilities.HARD_CODED_VARIABLES import UNIMODAL_MEAN


def uni_vs_bi_model_bics(data: list, std_dev, alpha):
    [x_data, y_data, params_uni, params_bi] = model_fitting(data, std_dev, alpha)

    unimodal_model = unimodal(x_data, UNIMODAL_MEAN, std_dev, *params_uni)
    residuals_uni = y_data - unimodal_model
    ss_res_uni = np.sum(residuals_uni ** 2)
    bic_uni = len(y_data) * np.log(ss_res_uni / len(y_data)) + len(params_uni) * np.log(len(y_data))

    bimodal_model = bimodal(x_data, alpha, std_dev, params_bi[0], params_bi[1])
    residuals_bi = y_data - bimodal_model
    ss_res_bi = np.sum(residuals_bi ** 2)
    bic_bi = len(y_data) * np.log(ss_res_bi / len(y_data)) + len(params_bi) * np.log(len(y_data))

    plot_models(x_data, y_data, unimodal_model, bimodal_model)

    return bic_uni, bic_bi
   