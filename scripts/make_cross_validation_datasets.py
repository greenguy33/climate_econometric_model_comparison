import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OrdinalEncoder
import pyfixest as pf

regression_data_full = pd.read_csv("../data/regression/gdp_regression_data.csv").dropna().reset_index(drop=True)

enc = OrdinalEncoder()

ordered_country_list = list(dict.fromkeys(regression_data_full.country))
enc.fit(np.array(ordered_country_list).reshape(-1,1))
regression_data_full["encoded_country"] = [int(val) for val in enc.transform(np.array(regression_data_full.country).reshape(-1,1))]

columns_to_center = list(regression_data_full.columns)[3:172]
# remove discrete natural disaster variables
del columns_to_center[1:7]

centered_data = pf.estimation.demean(
    np.array(regression_data_full[columns_to_center]), 
    np.array(regression_data_full[["year", "encoded_country"]]), 
    np.ones(len(regression_data_full))
)[0]

for index in range(len(columns_to_center)):
	columns_to_center[index] = columns_to_center[index] + "_centered"
centered_data = pd.DataFrame(centered_data, columns=columns_to_center)
centered_data = pd.concat([regression_data_full, centered_data], axis=1).reset_index(drop=True)

cv_folds = 10
sampled_years = np.array(random.sample(set(regression_data_full.year), k=len(set(regression_data_full.year))))
year_cut = OrdinalEncoder().fit_transform(np.array(list(pd.cut(range(1,len(set(regression_data_full.year))+1), bins=cv_folds))).reshape(-1,1)).flatten()

for fold in range(cv_folds):
    withheld_years = []
    for index, cut in enumerate(year_cut):
        if cut == fold:
            withheld_years.append(sampled_years[index])

    withheld_rows = centered_data.loc[(centered_data.year.isin(withheld_years))]
    training_rows = centered_data.loc[(centered_data.year.isin(withheld_years) == False)]
    
    training_rows.to_csv(f"../data/regression/cross_validation/gdp_regression_data_train_cv_{fold}.csv")
    withheld_rows.to_csv(f"../data/regression/cross_validation/gdp_regression_data_test_cv_{fold}.csv")