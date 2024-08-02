import pandas as pd
import numpy as np
import pickle as pkl
import pymc as pm
from pytensor import tensor as pt 
from sklearn.preprocessing import StandardScaler
import copy
import arviz as az


tfp_regression_data_insample = pd.read_csv("data/regression/cross_validation/tfp_regression_data_insample_festratified_0.csv")
countries_in_dataset = set(tfp_regression_data_insample.country)
years_in_dataset = set(tfp_regression_data_insample.year)
regions_in_dataset = set(tfp_regression_data_insample.region23)
country_fe_cols = [col for col in tfp_regression_data_insample.columns if "country_fixed_effect" in col]
country_ie_cols = [col for col in tfp_regression_data_insample.columns if "incremental_effect" in col]
year_fe_cols = [col for col in tfp_regression_data_insample.columns if "year_fixed_effect" in col]
region_fe_cols = [col for col in tfp_regression_data_insample.columns if "region_fixed_effect" in col]
for country_fe in country_fe_cols:
    country = country_fe.split("_")[0]
    if country not in countries_in_dataset:
        tfp_regression_data_insample = tfp_regression_data_insample.drop(country_fe, axis=1)
        for i in range(1,4):
            tfp_regression_data_insample = tfp_regression_data_insample.drop(country + f"_incremental_effect_{i}", axis=1)
for year_fe in year_fe_cols:
    year = year_fe.split("_")[0]
    if int(year) not in years_in_dataset:
        tfp_regression_data_insample = tfp_regression_data_insample.drop(year_fe, axis=1)
for region_fe in region_fe_cols:
    region = region_fe.split("_")[0]
    if region not in regions_in_dataset:
        tfp_regression_data_insample = tfp_regression_data_insample.drop(region_fe, axis=1)

model_spec = {
    "continuous_covariates" : [
        'temp_[weight]',
        'temp_[weight]_2', 
        'temp_[weight]_3'
    ],
    "discrete_covariates" : ['drought'],
    "fixed_effects" : ["year","region"],
    "incremental_effects" : 0,
    "weights" : "ag_weighted",
    "target" : "fd_ln_tfp"
}

covar_scalers = []
for covar_col in model_spec["continuous_covariates"]:
    covar_scalers.append(StandardScaler())
    tfp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])+"_scaled"] = covar_scalers[-1].fit_transform(np.array(tfp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])]).reshape(-1,1)).flatten()
target_var_scaler = StandardScaler()
tfp_regression_data_insample[model_spec["target"]+"_scaled"] = target_var_scaler.fit_transform(np.array(tfp_regression_data_insample[model_spec["target"]]).reshape(-1,1)).flatten()

target_data = tfp_regression_data_insample[model_spec["target"]+"_scaled"]
model_variables = []
for covar in model_spec["continuous_covariates"]:
    model_variables.append(covar.replace("[weight]",model_spec["weights"])+"_scaled")
for covar in model_spec["discrete_covariates"]:
    model_variables.append(covar)
for fe in model_spec["fixed_effects"]:
    for fe_col in [col for col in tfp_regression_data_insample.columns if col.endswith(f"{fe}_fixed_effect")]:
        model_variables.append(fe_col)
for i in range(model_spec["incremental_effects"]):
    for ie_col in [col for col in tfp_regression_data_insample.columns if col.endswith(f"incremental_effect_{i+1}")]:
        model_variables.append(ie_col)
model_data = tfp_regression_data_insample[model_variables]

first_year_fe_col = [col for col in model_data.columns if "year_fixed_effect" in col][0]

model_data_first_fe_removed = copy.deepcopy(model_data)
model_data_first_fe_removed[first_year_fe_col] = 0

vars_to_bundle = [col for col in model_data_first_fe_removed.columns if "year_fixed_effect" not in col and "region_fixed_effect" not in col and "incremental_effect" not in col and "country_fixed_effect" not in col and col != "drought"]
year_fe_vars = [col for col in model_data_first_fe_removed.columns if "year_fixed_effect" in col]
region_fe_vars = [col for col in model_data_first_fe_removed.columns if "region_fixed_effect" in col]
country_fe_vars = [col for col in model_data_first_fe_removed.columns if "country_fixed_effect" in col]

print(vars_to_bundle)

with pm.Model() as pymc_model:
    
    global_region_rs_mean = pm.Normal("global_region_rs_mean",0,1)
    global_region_rs_sd = pm.HalfNormal("global_region_rs_sd",1)
    region_rs_means = pm.Normal("region_rs_means", global_region_rs_mean, global_region_rs_sd, shape=(1,len(region_fe_vars)))
    region_rs_sd = pm.HalfNormal("region_rs_sd", 1)
    region_rs_coefs = pm.Normal("region_rs_coefs", region_rs_means, region_rs_sd)
    region_rs_matrix = pm.Deterministic("region_rs_matrix", pt.sum(region_rs_coefs * model_data[region_fe_vars],axis=1))

    drought_terms = pm.Deterministic("drought_terms", region_rs_matrix * model_data_first_fe_removed["drought"])

    covar_coefficients = pm.Normal("model_variable_coefs", 0, 5, shape=len(vars_to_bundle))
    covar_terms = pm.Deterministic("model_variable_terms", pt.sum(covar_coefficients * model_data_first_fe_removed[vars_to_bundle], axis=1))
    
    tfp_prior = pm.Deterministic(
        "tfp_prior",
        covar_terms +
        drought_terms
    )
    
    tfp_std_scale = pm.HalfNormal("tfp_std_scale", 10)
    tfp_std = pm.HalfNormal("tfp_std", sigma=tfp_std_scale)
    tfp_posterior = pm.Normal('tfp_posterior', mu=tfp_prior, sigma=tfp_std, observed=target_data)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

print(az.summary(trace, var_names=["region_rs_coefs"]))

with open ('output/models/bayes_models/tfp_random_no_year_intercept_random_region_drought_slope_model/tfp_random_no_year_intercept_random_region_drought_slope_model.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "var_list":list(model_data.columns),
        "model_spec":model_spec
    },buff)