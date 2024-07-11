import csv
import pandas as pd
import numpy as np
from countrycode import countrycode as cc
from calendar import monthrange
import warnings
from sklearn.preprocessing import OrdinalEncoder
import random
import collections
import itertools as it
from bisect import bisect_left
from sklearn.model_selection import StratifiedShuffleSplit

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def format_target_data(data, years, year_column_format, country_column, output_var):
	formatted_outcome_var = {}
	for row in data.iterrows():
		row = row[1]
		country = row[country_column]
		if country in iso3_countries_with_climate_data:
			formatted_outcome_var[country] = {}
			for year in years:
				year_data = float(row[year_column_format.replace("year",str(year))])
				last_year_data = float(row[year_column_format.replace("year",str(year-1))])
				formatted_outcome_var[country][year] = {}
				if np.isnan(year_data) or np.isnan(last_year_data):
					formatted_outcome_var[country][year][output_var] = np.NaN
				else:
					outcome = np.log(float(year_data)) - np.log(float(last_year_data))
					formatted_outcome_var[country][year][output_var] = outcome
	return formatted_outcome_var

def add_climate_vars_to_dataset(dataset, climate_val, prev_climate_val, country, year, climate_var, weights):
	dataset[country][year][f"{climate_var}_{weights}"] = climate_val
	dataset[country][year][f"{climate_var}_{weights}_2"] = np.square(climate_val)
	dataset[country][year][f"{climate_var}_{weights}_3"] = np.power(climate_val,3)
	if prev_climate_val != None:
		dataset[country][year][f"fd_{climate_var}_{weights}"] = climate_val - prev_climate_val
		dataset[country][year][f"fd_{climate_var}_{weights}_2"] = np.square(climate_val) - np.square(prev_climate_val)
		dataset[country][year][f"fd_{climate_var}_{weights}_3"] = np.power(climate_val,3) - np.power(prev_climate_val,3)
	return dataset

def add_natural_disasters_to_dataset(dataset, extracted_disasters, countries_with_natural_disaster_data):
	disaster_combinations = list(it.combinations(disaster_types_to_extract, 2))
	for country, data_by_year in dataset.items():
			for year, data in data_by_year.items():
				if country not in countries_with_natural_disaster_data:
					for disaster in disaster_types_to_extract:
						dataset[country][year][disaster] = np.NaN
					for disaster_combination in disaster_combinations:
						dataset[country][year][disaster_combination[0] + "_" + disaster_combination[1]] = np.NaN
				else:
					disaster_data = {}
					for disaster in disaster_types_to_extract:
						if country in extracted_disasters and year in extracted_disasters[country] and disaster in extracted_disasters[country][year]:
							disaster_data[disaster] = 1
						else:
							disaster_data[disaster] = 0
					for disaster_combination in disaster_combinations:
						if disaster_data[disaster_combination[0]] == 1 and disaster_data[disaster_combination[1]] == 1:
							disaster_data[disaster_combination[0] + "_" + disaster_combination[1]] = 1
						else:
							disaster_data[disaster_combination[0] + "_" + disaster_combination[1]] = 0
					for disaster, value in disaster_data.items():
						dataset[country][year][disaster] = value
	return dataset

def add_incremental_effects_to_dataset(file, year_range):
	dataset = pd.read_csv(file)
	for country in dataset.country:
		dataset[f"{country}_incremental_effect_1"] = np.where(dataset.country == country, 1, 0)
		dataset[f"{country}_incremental_effect_1"] = np.where(dataset.country==country, dataset[f"{country}_incremental_effect_1"].cumsum(), 0)
		dataset[f"{country}_incremental_effect_2"] = np.square(dataset[f"{country}_incremental_effect_1"])
		dataset[f"{country}_incremental_effect_3"] = np.power(dataset[f"{country}_incremental_effect_1"], 3)
	dataset.to_csv(file)

def add_fixed_effects_to_dataset(file):
	dataset = pd.read_csv(file)
	for country in sorted(list(set(dataset.country))):
		dataset[f"{country}_country_fixed_effect"] = np.where(dataset.country == country, 1, 0)
	for year in sorted(list(set(dataset.year))):
		dataset[f"{year}_year_fixed_effect"] = np.where(dataset.year == year, 1, 0)
	dataset.to_csv(file)

def write_regression_data_to_file(file, data):
	writer = csv.writer(file)
	headers =["country","year"]
	for column in data["AFG"][1962]:
		headers.append(column.replace(" ","_").lower())
	writer.writerow(headers)
	for country, data_by_year in dict(sorted(data.items(), key=lambda x: x[0])).items():
		for year, data in data_by_year.items():
			new_row = [country,year]
			for column in data:
				new_row.append(data[column])
			writer.writerow(new_row)

def find_closest_to_value_in_list(list_of_values, value_to_position, ):
	position = bisect_left(list_of_values, value_to_position)
	if position == 0:
		return position
	if position == len(list_of_values):
		return -1
	before = list_of_values[position - 1]
	after = list_of_values[position]
	if after - value_to_position < value_to_position - before:
		return position
	else:
		return position - 1

def create_target_distributed_test_and_training_datasets(data, target_var, nfolds=10):
	for i in range(nfolds):
		np.random.seed(i)
		data = data.dropna(axis=0).reset_index(drop=True)
		sorted_data = data.sort_values(by=target_var).reset_index(drop=True)
		target_var_data = list(sorted_data[target_var])
		num_test_samples = int(len(sorted_data)/10)
		samples = np.random.normal(np.mean(target_var_data), np.std(target_var_data), num_test_samples)
		indices_to_drop = []
		for sample in samples:
		    index = find_closest_to_value_in_list(target_var_data, sample)
		    indices_to_drop.append(index)
		    target_var_data[index] = np.NaN
		test_data = sorted_data.iloc[indices_to_drop]
		train_data = sorted_data.drop(index=indices_to_drop)
		train_data = train_data.reset_index(drop=True)

		enc = OrdinalEncoder()
		ordered_country_list = list(dict.fromkeys(train_data.country))
		enc.fit(np.array(ordered_country_list).reshape(-1,1))
		train_data["encoded_country_id"] = [int(val) for val in enc.transform(np.array(train_data.country).reshape(-1,1))]
		test_data["encoded_country_id"] = [int(val) for val in enc.transform(np.array(test_data.country).reshape(-1,1))]

		target_var_simple_name = target_var.split("_")[-1]
		train_data.sort_values(by=["country","year"]).to_csv(f"data/regression/cross_validation/{target_var_simple_name}_regression_data_insample_targetdistributed_{str(i)}.csv")
		test_data.sort_values(by=["country","year"]).to_csv(f"data/regression/cross_validation/{target_var_simple_name}_regression_data_outsample_targetdistributed_{str(i)}.csv")

def create_stratified_test_and_training_datasets(data, target_var, nfolds=10):
	target_var_simple_name = target_var.split("_")[-1]
	data = data.dropna(axis=0).reset_index(drop=True)
	splits = StratifiedShuffleSplit(n_splits=nfolds, test_size=.1).split(data, data.country + "_" + str(data.year))
	for nfold, (train_split, test_split) in enumerate(splits):
		training_rows = data.iloc[train_split]
		test_rows = data.iloc[test_split]
		training_rows.sort_values(by=["country","year"]).reset_index(drop=True).to_csv(f"data/regression/cross_validation/{target_var_simple_name}_regression_data_insample_festratified_{nfold}.csv")
		test_rows.sort_values(by=["country","year"]).reset_index(drop=True).to_csv(f"data/regression/cross_validation/{target_var_simple_name}_regression_data_outsample_festratified_{nfold}.csv")

gdp_data = pd.read_csv("data/GDP_per_capita/worldbank_wdi_gdp_per_capita.csv")
tfp_data = pd.read_csv("data/TFP/AgTFPInternational2021_AG_TFP.csv", header=2)
natural_disasters_data = pd.read_csv("data/natural_disasters/emdat_1960-2024.csv")
disaster_types_to_extract = ["Wildfire", "Drought", "Heat wave"]
gdp_years = range(1961,2024)
tfp_years = range(1962,2022)

fips_countries_with_climate_data = list(pd.read_csv("data/temp/monthly/processed_by_country/unweighted/temp.monthly.bycountry.unweighted.mean.csv").country)
iso3_country_list = cc(fips_countries_with_climate_data, origin="fips", destination="iso3c")
iso3_countries_with_climate_data = []
fips_indices_to_delete = []
for index, country in enumerate(iso3_country_list):
	if iso3_country_list[index] != None:
		iso3_countries_with_climate_data.append(country)
	else:
		fips_indices_to_delete.append(index)
fips_indices_to_delete = sorted(fips_indices_to_delete, reverse=True)
for index in fips_indices_to_delete:
	del fips_countries_with_climate_data[index]
assert len(fips_countries_with_climate_data) == len(iso3_countries_with_climate_data)
fips_to_iso3_country_map = {
	fips_countries_with_climate_data[index]:iso3_countries_with_climate_data[index]
	for index in range(len(iso3_countries_with_climate_data))
}

print("Adding GDP/TFP data...")

gdp_data = gdp_data.replace("..", np.NaN)
formatted_gdp_data = format_target_data(gdp_data, gdp_years, "year [YRyear]", "Country Code", "fd_ln_gdp")
formatted_tfp_data = format_target_data(tfp_data, tfp_years, "year", "ISO3", "fd_ln_tfp")

print("Adding natural disaster data...")

extracted_disasters = {}
missing_countries = set()
extreme_temp_subtypes = set()
countries_with_natural_disaster_data = set(natural_disasters_data.ISO)
for row in natural_disasters_data.iterrows():
	row = row[1]
	disaster_type = row["Disaster Type"]
	disaster_subtype = row["Disaster Subtype"]
	disaster = None
	if disaster_type in disaster_types_to_extract:
		disaster = disaster_type
	elif disaster_subtype in disaster_types_to_extract:
		disaster = disaster_subtype
	if disaster != None:
		country = row.ISO
		year = int(row["DisNo."].split("-")[0])
		if country not in extracted_disasters:
			extracted_disasters[country] = {}
		if year not in extracted_disasters[country]:
			extracted_disasters[country][year] = {}
		extracted_disasters[country][year][disaster] = 1

formatted_gdp_data = add_natural_disasters_to_dataset(formatted_gdp_data, extracted_disasters, countries_with_natural_disaster_data)
formatted_tfp_data = add_natural_disasters_to_dataset(formatted_tfp_data, extracted_disasters, countries_with_natural_disaster_data)

print("Adding monthly climate data...")

for climate_var in ["temp","precip","humidity"]:
	 for weights in ["unweighted", "pop_weighted","ag_weighted"]:
		 aggregate_var = "mean"
		 if weights != "unweighted":
			 aggregate_var = "weighted_mean"
		 weights_no_dash = weights.replace("_","")
		 data = pd.read_csv(f"data/{climate_var}/monthly/processed_by_country/{weights}/{climate_var}.monthly.bycountry.{weights_no_dash}.mean.csv")
		 for row in data.iterrows():
			 prev_climate_val = None
			 row = row[1]
			 if row.country in fips_to_iso3_country_map:
				 country = fips_to_iso3_country_map[row.country]
				 for year in range(1960,2024):
					 monthly_climate_vals = []
					 for month in range(1,13):
						 if month < 10:
							 month = "0" + str(month)
						 monthly_climate_vals.append(row[f"{weights_no_dash}_by_country.{aggregate_var}.X{year}.{month}.01"])
					 annual_climate_mean = np.mean(monthly_climate_vals)
					 if climate_var == "temp":
						 # celsius to kelvin
						 annual_climate_mean = annual_climate_mean - 273.15
					 elif climate_var == "precip":
						 # precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
						 annual_climate_mean = annual_climate_mean * 2.628e+6
					 if year in gdp_years and country in formatted_gdp_data:
						 formatted_gdp_data = add_climate_vars_to_dataset(formatted_gdp_data, annual_climate_mean, prev_climate_val, country, year, climate_var, weights)
					 if year in tfp_years and country in formatted_tfp_data:
						 formatted_tfp_data = add_climate_vars_to_dataset(formatted_tfp_data, annual_climate_mean, prev_climate_val, country, year, climate_var, weights)
					 prev_climate_val = annual_climate_mean

print("Adding daily climate data...(will take a few minutes)")

prev_results = {"annual_climate_std":{},"mean_daily_climate_std":{}}
for climate_var in ["temp","precip","humidity"]:
	 for weights in ["unweighted", "pop_weighted","ag_weighted"]:
		 aggregate_var = "mean"
		 if weights != "unweighted":
			 aggregate_var = "weighted_mean"
		 weights_no_dash = weights.replace("_","")
		 for year in range(1960,2024):
			 data = pd.read_csv(f"data/{climate_var}/daily/processed_by_country/{weights}/{climate_var}.daily.bycountry.{weights_no_dash}.{year}.csv")
			 data["ISO3"] = cc(data.country, origin="fips", destination="iso3c")
			 climate_columns = data.loc[:, data.columns.str.startswith(f"{weights_no_dash}_by_country.{aggregate_var}")]
			 data["annual_std"] = np.std(climate_columns, axis=1)
			 for measurement in range(0,1464,4):
				 data[f"daily_std_{int(measurement/4)}"] = np.std(climate_columns.iloc[:,measurement:measurement+4], axis=1)
			 data["mean_daily_std"] = np.mean(data.loc[:, data.columns.str.startswith("daily_std")], axis=1)
			 for row in data.iterrows():
				 row = row[1]
				 country = row.ISO3
				 mean_daily_climate_std = row.mean_daily_std
				 annual_climate_std = row.annual_std
				 prev_daily_std, prev_annual_std = None, None
				 if country in prev_results["mean_daily_climate_std"] and year-1 in prev_results["mean_daily_climate_std"][country]:
					 prev_daily_std = prev_results["mean_daily_climate_std"][country][year-1]
				 if country in prev_results["annual_climate_std"] and year-1 in prev_results["annual_climate_std"][country]:
					 prev_annual_std = prev_results["annual_climate_std"][country][year-1]
				 if year in gdp_years and country in formatted_gdp_data:
					 formatted_gdp_data = add_climate_vars_to_dataset(formatted_gdp_data, mean_daily_climate_std, prev_daily_std, country, year, climate_var + "_daily_std", weights)
					 formatted_gdp_data = add_climate_vars_to_dataset(formatted_gdp_data, annual_climate_std, prev_annual_std, country, year, climate_var + "_annual_std", weights)
				 if year in tfp_years and country in formatted_tfp_data:
					 formatted_tfp_data = add_climate_vars_to_dataset(formatted_tfp_data, mean_daily_climate_std, prev_daily_std, country, year, climate_var + "_daily_std", weights)
					 formatted_tfp_data = add_climate_vars_to_dataset(formatted_tfp_data, annual_climate_std, prev_annual_std, country, year, climate_var + "_annual_std", weights)
				 if country not in prev_results["mean_daily_climate_std"]:
					 prev_results["mean_daily_climate_std"][country] = {}
				 prev_results["mean_daily_climate_std"][country][year] = mean_daily_climate_std
				 if country not in prev_results["annual_climate_std"]:
					 prev_results["annual_climate_std"][country] = {}
				 prev_results["annual_climate_std"][country][year] = annual_climate_std

with open("data/regression/gdp_regression_data.csv", "w") as gdp_file:
	 write_regression_data_to_file(gdp_file, formatted_gdp_data)
with open("data/regression/tfp_regression_data.csv", "w") as tfp_file:
	 write_regression_data_to_file(tfp_file, formatted_tfp_data)

print("Adding incremental and fixed effects...")

add_incremental_effects_to_dataset("data/regression/gdp_regression_data.csv", gdp_years)
add_incremental_effects_to_dataset("data/regression/tfp_regression_data.csv", tfp_years)
add_fixed_effects_to_dataset("data/regression/gdp_regression_data.csv")
add_fixed_effects_to_dataset("data/regression/tfp_regression_data.csv")

print("Creating training/test splits...")

# create in-sample and out-of-sample datasets
gdp_data = pd.read_csv("data/regression/gdp_regression_data.csv")
tfp_data = pd.read_csv("data/regression/tfp_regression_data.csv")
create_target_distributed_test_and_training_datasets(gdp_data, "fd_ln_gdp")
create_target_distributed_test_and_training_datasets(tfp_data, "fd_ln_tfp")
create_stratified_test_and_training_datasets(gdp_data, "fd_ln_gdp")
create_stratified_test_and_training_datasets(tfp_data, "fd_ln_tfp")

print("Results written to data/regression")