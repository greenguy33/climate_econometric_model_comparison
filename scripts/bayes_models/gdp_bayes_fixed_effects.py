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

gdp_regression_data_insample = pd.read_csv("data/regression/cross_validation/gdp_regression_data_insample_festratified_0.csv")
countries_in_dataset = set(gdp_regression_data_insample.country)
years_in_dataset = set(gdp_regression_data_insample.year)
country_fe_cols = [col for col in gdp_regression_data_insample.columns if "country_fixed_effect" in col]
country_ie_cols = [col for col in gdp_regression_data_insample.columns if "incremental_effect" in col]
year_fe_cols = [col for col in gdp_regression_data_insample.columns if "year_fixed_effect" in col]
for country_fe in country_fe_cols:
    country = country_fe.split("_")[0]
    if country not in countries_in_dataset:
        gdp_regression_data_insample = gdp_regression_data_insample.drop(country_fe, axis=1)
        for i in range(1,4):
            gdp_regression_data_insample = gdp_regression_data_insample.drop(country + f"_incremental_effect_{i}", axis=1)
for year_fe in year_fe_cols:
    year = year_fe.split("_")[0]
    if int(year) not in years_in_dataset:
        gdp_regression_data_insample = gdp_regression_data_insample.drop(year_fe, axis=1)

model_spec = {
    "continuous_covariates" : [
        'fd_humidity_annual_std_[weight]', 'fd_humidity_annual_std_[weight]_2', 'fd_humidity_annual_std_[weight]_3', 
        'fd_humidity_daily_std_[weight]', 'fd_humidity_daily_std_[weight]_2', 'fd_precip_[weight]', 
        'fd_precip_[weight]_2', 'fd_precip_[weight]_3', 'fd_precip_annual_std_[weight]', 
        'fd_precip_annual_std_[weight]_2', 'fd_precip_daily_std_[weight]', 'fd_precip_daily_std_[weight]_2', 
        'fd_precip_daily_std_[weight]_3', 'fd_temp_annual_std_[weight]', 'fd_temp_annual_std_[weight]_2', 
        'fd_temp_annual_std_[weight]_3', 'precip_[weight]', 'precip_[weight]_2', 'precip_[weight]_3', 
        'precip_annual_std_[weight]', 'precip_annual_std_[weight]_2', 'precip_annual_std_[weight]_3', 
        'precip_daily_std_[weight]', 'precip_daily_std_[weight]_2', 'precip_daily_std_[weight]_3', 
        'temp_[weight]', 'temp_[weight]_2', 'temp_[weight]_3', 'temp_annual_std_[weight]', 
        'temp_annual_std_[weight]_2', 'temp_annual_std_[weight]_3', 'temp_daily_std_[weight]', 
        'temp_daily_std_[weight]_2', 'temp_daily_std_[weight]_3'
    ],
    "discrete_covariates" : ['drought', 'drought_heat_wave', 'wildfire_drought', 'wildfire_heat_wave'],
    "fixed_effects" : ["country","year"],
    "incremental_effects" : 3,
    "weights" : "unweighted",
    "target" : "fd_ln_gdp"
}

covar_scalers = []
for covar_col in model_spec["continuous_covariates"]:
    covar_scalers.append(StandardScaler())
    gdp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])+"_scaled"] = covar_scalers[-1].fit_transform(np.array(gdp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])]).reshape(-1,1)).flatten()
target_var_scaler = StandardScaler()
gdp_regression_data_insample[model_spec["target"]+"_scaled"] = target_var_scaler.fit_transform(np.array(gdp_regression_data_insample[model_spec["target"]]).reshape(-1,1)).flatten()

target_data = gdp_regression_data_insample[model_spec["target"]+"_scaled"]
model_variables = []
for covar in model_spec["continuous_covariates"]:
    model_variables.append(covar.replace("[weight]",model_spec["weights"])+"_scaled")
for covar in model_spec["discrete_covariates"]:
    model_variables.append(covar)
for fe in model_spec["fixed_effects"]:
    for fe_col in [col for col in gdp_regression_data_insample.columns if col.endswith(f"{fe}_fixed_effect")]:
        model_variables.append(fe_col)
for i in range(model_spec["incremental_effects"]):
    for ie_col in [col for col in gdp_regression_data_insample.columns if col.endswith(f"incremental_effect_{i+1}")]:
        model_variables.append(ie_col)
model_data = gdp_regression_data_insample[model_variables]

first_country_fe_col = [col for col in model_data.columns if "country_fixed_effect" in col][0]
first_year_fe_col = [col for col in model_data.columns if "year_fixed_effect" in col][0]

model_data_first_fe_removed = copy.deepcopy(model_data)
model_data_first_fe_removed[first_country_fe_col] = 0
model_data_first_fe_removed[first_year_fe_col] = 0

with pm.Model() as pymc_model:

    model_variable_coefs = pm.Normal("model_variable_coefs", 0, 10, shape=(len(model_data_first_fe_removed.columns)))
    model_terms = pm.Deterministic("model_variable_terms", pt.sum(model_variable_coefs * model_data_first_fe_removed, axis=1))

    gdp_std_scale = pm.HalfNormal("gdp_std_scale", 5)
    gdp_std = pm.HalfNormal("gdp_std", sigma=gdp_std_scale)
    gdp_posterior = pm.Normal('gdp_posterior', mu=model_terms, sigma=gdp_std, observed=target_data)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with open ('output/models/bayes_models/fixed_effects_model.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "variables_list":model_data_first_fe_removed.columns
    },buff)