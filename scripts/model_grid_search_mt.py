import csv
import pandas as pd
import numpy as np
import pyfixest as pf
import warnings
import copy
import concurrent.futures

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
		"out_sample_mse_reduction",
		"in_sample_mse_reduction",
		"out_sample_pred_int_acc",
		"in_sample_pred_int_acc"
	]

	def __init__(self):
		self.target_name = np.NaN
		self.model_vars = []
		self.out_sample_mse_reduction = np.NaN
		self.in_sample_mse_reduction = np.NaN
		self.out_sample_pred_int_acc = np.NaN
		self.in_sample_pred_int_acc = np.NaN
		self.fixed_effects = np.NaN
		self.incremental_effects = np.NaN
		self.weights = np.NaN

	def print_result(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val))

	def is_empty(self):
		if self.model_vars == []:
			return True
		else:
			return False

	def save_model_to_file(self):
		with open(f"output/models/{self.target_name}_best_model_from_grid_search_mt.csv", "w") as write_file:
			writer = csv.writer(write_file)
			for val in self.attrib_list:
				writer.writerow([val, getattr(self, val)])


def calculate_prediction_interval_accuracy(x, y, predictions, cov_mat):
	results = []
	for index, row in enumerate(x.itertuples()):
		x_data = list(row[1:])
		y_real = y.iloc[index]
		se_pred = np.sqrt(np.linalg.multi_dot([x_data, cov_mat, np.transpose(x_data)]))
		prediction_interval = (predictions[index]-se_pred*1.9603795, predictions[index]+se_pred*1.9603795)
		if y_real >= prediction_interval[0] and y_real <= prediction_interval[1]:
			results.append(1)
		else:
			results.append(0)
	return np.mean(results)


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
	
	in_sample_mse_list, out_sample_mse_list, in_sample_mse_red, out_sample_mse_red, in_sample_pred_int_acc, out_sample_pred_int_acc = [], [], [], [], [], []
	model_vars_with_weights = [var.replace("[weight]",weights) for var in model_vars]
	print(model_vars_with_weights)

	data_columns = pd.read_csv(f"data/regression/cross_validation/{target_name}_regression_data_insample_0.csv").columns
	if incremental_effects != 0:
		for i in range(incremental_effects):
			for incremental_col in [col for col in data_columns if col.endswith(f"incremental_effect_{str(i+1)}")]:
				model_vars_with_weights.append(incremental_col)
	
	for fold in range(num_folds):
		
		train_data = train_data_files[fold]
		test_data = test_data_files[fold]

		train_data_covariates = train_data[model_vars_with_weights]
		test_data_covariates = test_data[model_vars_with_weights]
		covariate_string = " + ".join(model_vars_with_weights)
		
		regression = pf.feols(
			f"{target_var} ~ {covariate_string} | {fixed_effects}", 
			data=train_data
		)
		only_intercept_regression = pf.feols(
			f"{target_var} ~ 1 | 0", 
			data=train_data
		)

		# remove variables from sample data that were removed from regression due to multicollinearity
		vars_in_model = regression._coefnames
		for column in list(train_data_covariates.columns):
			if column not in vars_in_model:
				train_data_covariates.drop(column, axis=1, inplace=True)
				test_data_covariates.drop(column, axis=1, inplace=True)
		
		cov_mat = pd.DataFrame(regression._vcov)

		# remove intercept from non-fixed-effects covariance matrix
		if fixed_effects == 0:
			cov_mat= cov_mat.drop(0, axis=1).drop(0, axis=0)

		in_sample_predictions = regression.predict(train_data)
		intercept_only_in_sample_predictions = only_intercept_regression.predict(train_data)
		in_sample_mse = np.mean(np.square(in_sample_predictions-train_data[target_var]))
		intercept_only_in_sample_mse = np.mean(np.square(intercept_only_in_sample_predictions-train_data[target_var]))
		in_sample_mse_red.append((intercept_only_in_sample_mse-in_sample_mse)/intercept_only_in_sample_mse*100)
		in_sample_mse_list.append(in_sample_mse)
		in_sample_pred_int_acc.append(calculate_prediction_interval_accuracy(train_data_covariates, train_data[target_var], in_sample_predictions, cov_mat))

		out_sample_predictions = regression.predict(test_data)
		intercept_only_out_sample_predictions = only_intercept_regression.predict(test_data)
		out_sample_mse = np.mean(np.square(out_sample_predictions-test_data[target_var]))
		intercept_only_out_sample_mse = np.mean(np.square(intercept_only_out_sample_predictions-test_data[target_var]))
		out_sample_mse_red.append((intercept_only_out_sample_mse-out_sample_mse)/intercept_only_out_sample_mse*100)
		out_sample_mse_list.append(out_sample_mse)
		out_sample_pred_int_acc.append(calculate_prediction_interval_accuracy(test_data_covariates, test_data[target_var], out_sample_predictions, cov_mat))
	
	reg_result = RegressionResult()
	reg_result.target_name = target_name
	reg_result.weights = weights
	reg_result.model_vars = sorted(model_vars)
	reg_result.in_sample_mse = np.mean(in_sample_mse_list)
	reg_result.out_sample_mse = np.mean(out_sample_mse_list)
	reg_result.in_sample_mse_reduction = np.mean(in_sample_mse_red)
	reg_result.out_sample_mse_reduction = np.mean(out_sample_mse_red)
	reg_result.in_sample_pred_int_acc = np.mean(in_sample_pred_int_acc)
	reg_result.out_sample_pred_int_acc = np.mean(out_sample_pred_int_acc)
	reg_result.fixed_effects = fixed_effects
	reg_result.incremental_effects = incremental_effects
	return reg_result


def find_best_model(args, num_folds=1):

	fe, ie, weights = args[0], args[1], args[2]
	print("fixed effects:", fe, "incremental_effects:", ie, "weights:", weights)

	base_model = RegressionResult()
	for group, vars in model_variations.items():
		model_vars = []
		for var in vars:
			model_vars.append(var)
			new_model = run_fe_regression_with_cv(num_folds, target_name, target_var, weights, model_vars, fe, ie)
			if base_model.is_empty():
				base_model = new_model
			else:
				base_model = choose_best_model(base_model, new_model, stat="mse")

	model_vars_to_add = set()
	for group, vars in model_variations.items():
		new_model_vars = []
		for model_var in base_model.model_vars:
			new_model_vars.append(model_var)
		for var in vars:
			if var not in base_model.model_vars:
				new_model_vars.append(var)
				new_model = run_fe_regression_with_cv(num_folds, target_name, target_var, weights, new_model_vars, fe, ie)
				if choose_best_model(base_model, new_model, stat="mse") == new_model:
					for model_var in new_model_vars:
						if model_var not in base_model.model_vars:
							model_vars_to_add.add(model_var)

	second_round_model_vars = []
	for var in base_model.model_vars:
		second_round_model_vars.append(var)
	for var in model_vars_to_add:
		second_round_model_vars.append(var)
	second_round_model_vars = list(set(second_round_model_vars))
	new_model = run_fe_regression_with_cv(num_folds, target_name, target_var, weights, second_round_model_vars, fe, ie)
	new_model = choose_best_model(base_model, new_model, stat="mse")
	
	return new_model


# load training and test data into memory
train_data_files = {}
test_data_files = {}
for i in range(10):
	train_data_files[i] = pd.read_csv(f"data/regression/cross_validation/gdp_regression_data_insample_{str(i)}.csv")
	test_data_files[i] = pd.read_csv(f"data/regression/cross_validation/gdp_regression_data_outsample_{str(i)}.csv")


model_variations = {
	"temp_vars":["temp_[weight]","temp_[weight]_2"]#,"temp_[weight]_3"],
	# "precip_vars":["precip_[weight]","precip_[weight]_2","precip_[weight]_3"],
	# "humidity_vars":["humidity_[weight]","humidity_[weight]_2","humidity_[weight]_3"],
	# "temp_daily_std_vars":["temp_daily_std_[weight]","temp_daily_std_[weight]_2","temp_daily_std_[weight]_3"],
	# "precip_daily_std_vars":["precip_daily_std_[weight]","precip_daily_std_[weight]_2","precip_daily_std_[weight]_3"],
	# "humidity_daily_std_vars":["humidity_daily_std_[weight]","humidity_daily_std_[weight]_2","humidity_daily_std_[weight]_3"],
	# "temp_annual_std_vars":["temp_annual_std_[weight]","temp_annual_std_[weight]_2","temp_annual_std_[weight]_3"],
	# "precip_annual_std_vars":["precip_annual_std_[weight]","precip_annual_std_[weight]_2","precip_annual_std_[weight]_3"],
	# "humidity_annual_std_vars":["humidity_annual_std_[weight]","humidity_annual_std_[weight]_2","humidity_annual_std_[weight]_3"],
	# "fd_temp_vars":["fd_temp_[weight]","fd_temp_[weight]_2","fd_temp_[weight]_3"],
	# "fd_precip_vars":["fd_precip_[weight]","fd_precip_[weight]_2","fd_precip_[weight]_3"],
	# "fd_humidity_vars":["fd_humidity_[weight]","fd_humidity_[weight]_2","fd_humidity_[weight]_3"],
	# "fd_temp_daily_std_vars":["fd_temp_daily_std_[weight]","fd_temp_daily_std_[weight]_2","fd_temp_daily_std_[weight]_3"],
	# "fd_precip_daily_std_vars":["fd_precip_daily_std_[weight]","fd_precip_daily_std_[weight]_2","fd_precip_daily_std_[weight]_3"],
	# "fd_humidity_daily_std_vars":["fd_humidity_daily_std_[weight]","fd_humidity_daily_std_[weight]_2","fd_humidity_daily_std_[weight]_3"],
	# "fd_temp_annual_std_vars":["fd_temp_annual_std_[weight]","fd_temp_annual_std_[weight]_2","fd_temp_annual_std_[weight]_3"],
	# "fd_precip_annual_std_vars":["fd_precip_annual_std_[weight]","fd_precip_annual_std_[weight]_2","fd_precip_annual_std_[weight]_3"],
	# "fd_humidity_annual_std_vars":["fd_humidity_annual_std_[weight]","fd_humidity_annual_std_[weight]_2","fd_humidity_annual_std_[weight]_3"],
	# "drought":["drought"],
	# "wildfire":["wildfire"],
	# "heat_wave":["heat_wave"],
	# "wildfire_drought":["wildfire_drought"],
	# "wildfire_heat_wave":["wildfire_heat_wave"],
	# "drought_heat_wave":["drought_heat_wave"]
}

target_var_list = {
	"gdp":"fd_ln_gdp",
	# "tfp":"fd_ln_tfp"
}

effect_variations = {
	"fixed_effects":[0,"year","country"],#,"year","country + year"],
	"incremental_effects":[0,2,3],
	"weights":["unweighted","pop_weighted","ag_weighted"]
}

import time
starttime = time.time()
for target_name, target_var in target_var_list.items():
	print(target_name, target_var)

	with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
		models = executor.map(find_best_model, ([fe, ie, weights] for fe in effect_variations["fixed_effects"] for ie in effect_variations["incremental_effects"] for weights in effect_variations["weights"]))

	for model in models:
		model.print_result()

	overall_best_model = RegressionResult()
	for new_model in models:
		if overall_best_model.is_empty():
			overall_best_model = new_model
		else:
			overall_best_model = choose_best_model(overall_best_model, new_model, stat="mse")

	overall_best_model.print_result()
	overall_best_model.save_model_to_file()
endtime = time.time()
print(endtime-starttime)