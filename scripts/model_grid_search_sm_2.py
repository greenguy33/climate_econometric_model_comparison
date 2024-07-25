import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import copy
from multiprocessing.pool import ThreadPool

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')


class RegressionResult:

	attrib_list = [
		"model_vars",
		"fixed_effects",
		"incremental_effects",
		"weights",
		"out_sample_mse",
		"in_sample_mse",
		"out_sample_pred_int_acc",
		"in_sample_pred_int_acc"
	]

	def __init__(self):
		self.target_name = np.NaN
		self.model_vars = []
		self.out_sample_pred_int_acc = np.NaN
		self.in_sample_pred_int_acc = np.NaN
		self.fixed_effects = np.NaN
		self.incremental_effects = np.NaN
		self.weights = np.NaN

	def print_result(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val), flush=True)

	def is_empty(self):
		if self.model_vars == []:
			return True
		else:
			return False

	def save_model_to_file(self):
		with open(f"output/models/{self.target_name}_best_model_from_grid_search_modified.csv", "w") as write_file:
			writer = csv.writer(write_file)
			for val in self.attrib_list:
				writer.writerow([val, getattr(self, val)])


def calculate_prediction_interval_accuracy(y, predictions):
	pred_data = pd.DataFrame(np.transpose([y, predictions.predicted_mean, predictions.var_pred_mean]), columns=["real_y", "pred_mean", "pred_var"])
	pred_data["pred_int_acc"] = np.where(
		(pred_data.pred_mean + np.sqrt(pred_data.pred_var) * 1.9603795 > pred_data.real_y) &
		(pred_data.pred_mean - np.sqrt(pred_data.pred_var) * 1.9603795 < pred_data.real_y),
		1,
		0
	)
	return np.mean(pred_data.pred_int_acc)


def choose_best_model(model1, model2, stat="mse"):
	assert stat in ["pred_int","mse","pred_int+mse"]
	if stat == "pred_int+mse":
		if (model1.out_sample_mse > model2.out_sample_mse) and (abs(.95-model1.out_sample_pred_int_acc) > abs(.95-model2.out_sample_pred_int_acc)):
			return model2
		else:
			return model1
	elif stat == "pred_int":
		if (abs(.95-model1.out_sample_pred_int_acc) > abs(.95-model2.out_sample_pred_int_acc)):
			return model2
		else:
			return model1
	elif stat == "mse":
		if (model1.out_sample_mse > model2.out_sample_mse):
			return model2
		else:
			return model1


def run_fe_regression_with_cv(num_folds, target_name, target_var, weights, model_vars, fixed_effects, incremental_effects):
	
	in_sample_mse_list, out_sample_mse_list, in_sample_pred_int_acc, out_sample_pred_int_acc = [], [], [], []
	model_vars_with_weights = [var.replace("[weight]",weights) for var in model_vars]
	print(model_vars_with_weights, flush=True)

	data_columns = train_data_files[0].columns

	if incremental_effects != 0:
		for i in range(incremental_effects):
			for incremental_col in [col for col in data_columns if col.endswith(f"incremental_effect_{str(i+1)}")]:
				model_vars_with_weights.append(incremental_col)

	if fixed_effects != None:
		for fe in fixed_effects:
			for fe_col in [col for col in data_columns if col.endswith(f"{fe}_fixed_effect")]:
				model_vars_with_weights.append(fe_col)

	for fold in range(num_folds):
		train_data = train_data_files[fold]
		test_data = test_data_files[fold]

		train_data_covariates = train_data[model_vars_with_weights]
		test_data_covariates = test_data[model_vars_with_weights]

		if fixed_effects == None:
			train_data_covariates = sm.add_constant(train_data_covariates)
			test_data_covariates = sm.add_constant(test_data_covariates)

		model = sm.OLS(train_data[target_var],train_data_covariates,missing="drop")
		try:
			regression = model.fit()
		except np.linalg.LinAlgError:
			weights_as_string = ",".join(model_vars)
			print(f"Skipping model with variables {weights_as_string} - regression failed to converge")
			reg_result = RegressionResult()
			reg_result.out_sample_mse = 999
			reg_result.out_sample_pred_int_acc = 0
			return reg_result

		in_sample_predictions = regression.get_prediction(train_data_covariates)
		out_sample_predictions = regression.get_prediction(test_data_covariates)

		in_sample_mse_list.append(np.mean(np.square(in_sample_predictions.predicted_mean-train_data[target_var])))
		out_sample_mse_list.append(np.mean(np.square(out_sample_predictions.predicted_mean-test_data[target_var])))

		in_sample_pred_int_acc.append(calculate_prediction_interval_accuracy(train_data[target_var], in_sample_predictions))
		out_sample_pred_int_acc.append(calculate_prediction_interval_accuracy(test_data[target_var], out_sample_predictions))
	
	reg_result = RegressionResult()
	reg_result.target_name = target_name
	reg_result.weights = weights
	reg_result.model_vars = sorted(model_vars)
	reg_result.in_sample_mse = np.mean(in_sample_mse_list)
	reg_result.out_sample_mse = np.mean(out_sample_mse_list)
	reg_result.in_sample_pred_int_acc = np.mean(in_sample_pred_int_acc)
	reg_result.out_sample_pred_int_acc = np.mean(out_sample_pred_int_acc)
	reg_result.fixed_effects = fixed_effects
	reg_result.incremental_effects = incremental_effects
	return reg_result


def find_best_model(target_name, target_var, num_folds, fe, ie, weights):

	print("fixed effects:", fe, "incremental_effects:", ie, "weights:", weights, flush=True)

	base_model = run_fe_regression_with_cv(num_folds, target_name, target_var, weights, base_model_vars, fe, ie)
	best_model = copy.deepcopy(base_model)

	model_vars_to_add = set()
	for var_group in model_variations:
		new_model_vars = []
		for model_var in base_model.model_vars:
			new_model_vars.append(model_var)
		for var in var_group:
			if var not in base_model.model_vars:
				new_model_vars.append(var)
				new_model = run_fe_regression_with_cv(num_folds, target_name, target_var, weights, new_model_vars, fe, ie)
				best_model = choose_best_model(best_model, new_model, stat="mse")

	return best_model


train_data_files, test_data_files = {}, {}

base_model_vars = ["temp_[weight]","temp_[weight]_2","temp_[weight]_3","drought"]

model_variations = [
	["precip_[weight]","precip_[weight]_2","precip_[weight]_3"],
	["humidity_[weight]","humidity_[weight]_2","humidity_[weight]_3"],
	["temp_daily_std_[weight]","temp_daily_std_[weight]_2","temp_daily_std_[weight]_3"],
	["precip_daily_std_[weight]","precip_daily_std_[weight]_2","precip_daily_std_[weight]_3"],
	["humidity_daily_std_[weight]","humidity_daily_std_[weight]_2","humidity_daily_std_[weight]_3"],
	["temp_annual_std_[weight]","temp_annual_std_[weight]_2","temp_annual_std_[weight]_3"],
	["precip_annual_std_[weight]","precip_annual_std_[weight]_2","precip_annual_std_[weight]_3"],
	["humidity_annual_std_[weight]","humidity_annual_std_[weight]_2","humidity_annual_std_[weight]_3"],
	["fd_temp_[weight]","fd_temp_[weight]_2","fd_temp_[weight]_3"],
	["fd_precip_[weight]","fd_precip_[weight]_2","fd_precip_[weight]_3"],
	["fd_humidity_[weight]","fd_humidity_[weight]_2","fd_humidity_[weight]_3"],
	["fd_temp_daily_std_[weight]","fd_temp_daily_std_[weight]_2","fd_temp_daily_std_[weight]_3"],
	["fd_precip_daily_std_[weight]","fd_precip_daily_std_[weight]_2","fd_precip_daily_std_[weight]_3"],
	["fd_humidity_daily_std_[weight]","fd_humidity_daily_std_[weight]_2","fd_humidity_daily_std_[weight]_3"],
	["fd_temp_annual_std_[weight]","fd_temp_annual_std_[weight]_2","fd_temp_annual_std_[weight]_3"],
	["fd_precip_annual_std_[weight]","fd_precip_annual_std_[weight]_2","fd_precip_annual_std_[weight]_3"],
	["fd_humidity_annual_std_[weight]","fd_humidity_annual_std_[weight]_2","fd_humidity_annual_std_[weight]_3"],
]

target_var_list = {
	"gdp":"fd_ln_gdp",
	# "tfp":"fd_ln_tfp"
}

effect_variations = {
	"fixed_effects":[["country","year"],["year"],["country"],None],
	"weights":["pop_weighted","unweighted"]
}

num_folds=10

import time
starttime = time.time()
for target_name, target_var in target_var_list.items():
	print(target_name, target_var, flush=True)

	# load training and test data into memory
	for i in range(num_folds):
		train_data_files[i] = pd.read_csv(f"data/regression/cross_validation/{target_name}_regression_data_insample_festratified_{str(i)}.csv")
		test_data_files[i] = pd.read_csv(f"data/regression/cross_validation/{target_name}_regression_data_outsample_festratified_{str(i)}.csv")

	models = []
	for fe in effect_variations["fixed_effects"]:
		for weights in effect_variations["weights"]:
			models.append(find_best_model(target_name, target_var, num_folds, fe, 0, weights))

	overall_best_model = RegressionResult()
	for new_model in models:
		if overall_best_model.is_empty():
			overall_best_model = new_model
		else:
			overall_best_model = choose_best_model(overall_best_model, new_model, stat="mse")

	overall_best_model.print_result()
	overall_best_model.save_model_to_file()

	train_data_files, test_data_files = {}, {}
endtime = time.time()
print(endtime-starttime, flush=True)