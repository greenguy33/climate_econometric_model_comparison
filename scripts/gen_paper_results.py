import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import geopandas
import folium
import seaborn as sns

data = pd.read_csv("data/regression/tfp_regression_data.csv").dropna().reset_index(drop=True)

model0_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_withheld_10k_no_temp_only_country_coefs.pkl'
model1_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_full_10k_temp_only_country_coefs.pkl'
model2_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_withheld_10k_temp_and_precip_only_country_coefs.pkl'

model0 = pd.read_pickle(model0_file)
model1 = pd.read_pickle(model1_file)
model2 = pd.read_pickle(model2_file)

# add total droughts by country to dataset
droughts_by_country = {}
for country in set(data.country):
    droughts_by_country[country] = np.count_nonzero(data.loc[(data.country == country)].drought)
data["total_drought_by_country"] = list(map(lambda x : droughts_by_country[x], data.country))

# find countries that have no drought in dataset
countries_with_no_drought = []
for country in set(data.country):
    if all(data[data.country == country].drought == 0):
        countries_with_no_drought.append(country)

# import ag. revenue share data
revenue_data = pd.read_csv("data/revenue_shares.csv")
country_weights = {}
for row in revenue_data.itertuples():
    if row[3] in set(data.country):
        country_weights[row[3]] = np.mean([row[5],row[6],row[7],row[8],row[9],row[10]])
weight_sum = sum(list(country_weights.values()))
for country, val in country_weights.items():
    country_weights[country] = val/weight_sum

# import developed vs. developing country data
country_development_classification = pd.read_csv("data/developed_developing_countries_UN.csv")
country_development_classification = {row[1]["ISO-alpha3 Code"]:row[1]["Developed / Developing regions"] for row in country_development_classification.iterrows()}
# removing Taiwan because it is not in country development data
data_mod = data[~data.country.isin(["TWN"])]
development_classification = []
for row in data_mod.iterrows():
    row = row[1]
    development_classification.append(country_development_classification[row.country])
data_mod["development"] = development_classification

model_impacts = {}
for model_index, model in enumerate([model0, model1, model2]):

    # unscale country coefficients
    scaled_vars = {}
    unscaled_vars = {}
    for country_index, var in enumerate(model["var_list"][-163:]):
        scaled_vars[var] = model["posterior"][:,:,:,country_index].data.flatten()
    for var, samples in scaled_vars.items():
        unscaled_vars[var] = np.array(samples) * np.std(data.fd_ln_tfp)

    # compute probability that drought has decreased TFP for each country
    country_percentiles = {}
    for country in list(model["var_list"][-163:]):
        country_percentiles[country.split("_")[0]] = len([sample for sample in unscaled_vars[country] if sample < 0])/len(unscaled_vars[country])

    # compute % impacts by country
    effect_by_country = {}
    for country in set(data.country):
        effect_by_country[country] = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for i in range(droughts_by_country[country]):
            effect_by_country[country] += unscaled_vars[country+"_country_fixed_effect"]
    percent_loss_by_country = {
        country:np.array([math.expm1(val)*100 for val in effect_by_country[country]])
        for country in set(data.country)
    }

    # compute % impacts by region
    percent_loss_by_region = {}
    for region in set(data.region23):
        countries = set(data.loc[data.region23 == region].country)
        sum_of_group_weights = 0
        for country in countries:
            sum_of_group_weights += country_weights[country]
        group_country_weights = {country:country_weights[country]/sum_of_group_weights for country in countries}
        region_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for country in countries:
            country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
            for i in range(droughts_by_country[country]):
                country_effect += unscaled_vars[country+"_country_fixed_effect"] * group_country_weights[country]
            if not all(country_effect) == 0:
                region_effect += country_effect
        percent_loss_by_region[region] = [math.expm1(val)*100 for val in region_effect]

    # compute probability that drought has decreased TFP for each region
    region_percentiles = {}
    for region in set(data.region23):
        region_percentiles[region] = len([val for val in percent_loss_by_region[region] if val < 0])/20000

    # compute global % impact
    global_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
    for country in set(data.country):
        country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for i in range(droughts_by_country[country]):
            country_effect += unscaled_vars[country+"_country_fixed_effect"] * country_weights[country]
        for i, val in enumerate(country_effect):
            global_effect[i] += val
    global_effect_percent = [math.expm1(val)*100 for val in global_effect]

    # compute % impacts for developed vs. developing countries
    development_effect_percent = {}
    for development in ["Developing","Developed"]:
        countries = [country for country in set(data_mod.country) if country_development_classification[country] == development]
        sum_of_group_weights = 0
        for country in countries:
            sum_of_group_weights += country_weights[country]
        group_country_weights = {country:country_weights[country]/sum_of_group_weights for country in countries}
        development_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for country in countries:
            country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
            for i in range(droughts_by_country[country]):
                country_effect += unscaled_vars[country+"_country_fixed_effect"] * group_country_weights[country]
            for i, val in enumerate(country_effect):
                development_effect[i] += val
        development_effect_percent[development] = [math.expm1(val)*100 for val in development_effect]

    # add model results to dict
    model_results = {
        "country_percentiles":country_percentiles,
        "percent_loss_by_country":percent_loss_by_country,
        "region_percentiles":region_percentiles,
        "percent_loss_by_region":percent_loss_by_region,
        "global_percent_loss":global_effect_percent,
        "development_percent_loss":development_effect_percent
    }
    model_impacts[f"model{model_index}"] = model_results

print(model_impacts.keys())

# choose model to use to generate figures
model = "model1"
model_data = model_impacts[model]

# Plot Figure 1`
fig, axes = plt.subplots(1,2,figsize=(15,15))
country_geopandas = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
country_geopandas = country_geopandas.merge(
    data,
    how='inner', 
    left_on=['iso_a3'],
    right_on=['country']
)
res = country_geopandas.plot(column="total_drought_by_country", cmap="RdYlGn_r", ax=axes[0], legend=True, legend_kwds={"location":"bottom", "pad":.04})
axes[0].set_title("Total Droughts by Country (1961 - 2021)", size=20, weight="bold")
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])
axes[0].set_xlabel("Number of Droughts", size=20, weight="bold")
axes[0].figure.axes[1].tick_params(labelsize=50)
res.figure.axes[-1].tick_params(labelsize=15)

country_geopandas = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
tfp_regression_data_ndcr = data[~data.country.isin(countries_with_no_drought)]
country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axes[1])
country_geopandas = country_geopandas.merge(
    tfp_regression_data_ndcr,
    how='inner',
    left_on=['iso_a3'],
    right_on=['country']
)
res = country_geopandas.plot(column="drought_bin", cmap="RdYlGn_r", ax=axes[1], legend=True, legend_kwds={"location":"bottom", "pad":.04})

cmap = plt.get_cmap('RdYlGn_r')
norm = mcolors.Normalize(vmin=country_geopandas['drought_bin'].min(), vmax=country_geopandas['drought_bin'].max())
colors = [cmap(norm(value)) for value in country_geopandas['drought_bin']]
axes[1].set_title("Country Prob. that drought has \n decreased TFP (Map)", size=20, weight="bold")
axes[1].set_xlabel("Percentage (%)", size=20, weight="bold")
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])
res.figure.axes[-1].tick_params(labelsize=15)

plt.savefig("figures/drought_fig1.png", bbox_inches='tight')