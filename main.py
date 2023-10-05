from config.config import FINAL_HEADING_DATA
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn import mixture
import pandas as pd
from utilities.models import unimodal, bimodal
from utilities.uni_vs_bi_model_bics import uni_vs_bi_model_bics

df = pd.read_csv(FINAL_HEADING_DATA)

ALPHA_CONDITION = float(120) # the condition that we're assessing
alpha_heading = ALPHA_CONDITION/2 # the heading that the subgroups are moving towards
ALPHA_INPUT = float(0) # the condition that is being used to create the statistical model

assessed_condition = df[df['Alpha'] == ALPHA_CONDITION]['Final_Heading'].tolist()
input_condition = df[df['Alpha'] == ALPHA_INPUT]['Final_Heading'].tolist()

mean = np.mean(input_condition)
std_dev = np.std(input_condition)


[ bic_uni, bic_bi ] = uni_vs_bi_model_bics(assessed_condition, mean, std_dev, alpha_heading)
print("BIC Unimodal:", bic_uni)
print("BIC Bimodal:", bic_bi)
f = 0
