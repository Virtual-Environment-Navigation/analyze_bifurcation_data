from config.config import FINAL_HEADING_DATA
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn import mixture
import pandas as pd

df = pd.read_csv(FINAL_HEADING_DATA)
