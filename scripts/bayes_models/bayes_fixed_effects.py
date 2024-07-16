import csv
import os
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle as pkl
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
import warnings
import pymc as pm
from pytensor import tensor as pt 

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')

gdp_regression_data = pd.read_csv("../data/regression/cross_validation/gdp_regression_data_insample_festratified_0.csv")

model_spec = {
    "continuous_covariates" : [
        'fd_humidity_[weight]', 'fd_humidity_[weight]_2', 'fd_humidity_annual_std_[weight]', 
        'fd_humidity_annual_std_[weight]_2', 'fd_precip_[weight]', 'fd_precip_[weight]_2', 
        'fd_precip_[weight]_3', 'fd_precip_annual_std_[weight]', 'fd_precip_daily_std_[weight]', 
        'fd_precip_daily_std_[weight]_2', 'fd_precip_daily_std_[weight]_3', 'fd_temp_annual_std_[weight]', 
        'fd_temp_annual_std_[weight]_2', 'fd_temp_annual_std_[weight]_3', 'fd_temp_daily_std_[weight]', 
        'fd_temp_daily_std_[weight]_2', 'humidity_daily_std_[weight]', 'precip_[weight]', 
        'precip_daily_std_[weight]', 'temp_[weight]', 'temp_[weight]_2', 
        'temp_[weight]_3'
    ],
    "discrete_covariates" : ['wildfire','drought','wildfire_drought'],
    "fixed_effects" : ["country","year"],
    "incremental_effects" : 3,
    "weights" : "unweighted",
    "target" : "fd_ln_gdp"
}

for covar_col in model_spec["continuous_covariates"]:
    covar_scalers.append(StandardScaler())
    gdp_regression_data[covar_col.replace("[weight]",model_spec["weights"])+"_scaled"] = covar_scalers[-1].fit_transform(np.array(gdp_regression_data[covar_col.replace("[weight]",model_spec["weights"])]).reshape(-1,1)).flatten()
target_var_scaler = StandardScaler()
gdp_regression_data[model_spec["target"]+"_scaled"] = target_var_scaler.fit_transform(np.array(gdp_regression_data[model_spec["target"]]).reshape(-1,1)).flatten()

