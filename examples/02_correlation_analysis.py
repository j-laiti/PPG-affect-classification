# The purpose of this file is to create correlation graphs between HRV features and stress/fatigue labels
# Figure 4 in the manuscript was created using this code.
# Justin Laiti June 23 2025

### WESAD Correlation Analysis ###
  

# TODO: come back to this later to test different datasets to see which one was included in the paper

#%% imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import numpy as np

#%%

# Load both CSV files
wesad_features_file_path = "../data/WESAD/data_merged_bp_time.csv"

# Load datasets & add a column indicating the processing pipeline
df2 = pd.read_csv(wesad_features_file_path)

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50']
label = ['label']

#%% run correlation analysis for stress labels

# isolate features of interest
time_domain_features = df2[feats + label]

# arrays to save correlation results
wesad_correlations = []
feature_names = []

print("Point Biserial Correlation Analysis:")
print("====================================")

for feature in feats:
    corr, p_value = pointbiserialr(time_domain_features[feature], time_domain_features["label"])
    wesad_correlations.append(corr)
    feature_names.append(feature)

    print(f"{feature}: Correlation = {corr:.2f}, p-value = {p_value:.3f}")

#%% Create correlation heatmap

correlation_data = pd.DataFrame({
    'Stress Correlation': wesad_correlations,
}, index=feature_names)

# Plot heatmap
plt.figure(figsize=(4, 8))
sns.heatmap(correlation_data, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,       # Center colormap at 0
            cbar_kws={'label': 'Correlation with Stress'},
            linewidths=0.5)

plt.title('HRV Features Correlation with Stress Classification\n(WESAD Dataset)', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('HRV Features', fontsize=12)
plt.xlabel('Stress Correlation', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()






# %%
# ### Wellby Correlation Analysis ###

# Load both CSV files
wellby_features_file_path = "../data/Wellby/combined_sim_features_adj4.csv"

# Load datasets & add a column indicating the processing pipeline
df = pd.read_csv(wellby_features_file_path)

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50']
labels = ['sleep_binary','stress_binary']

df2 = df[feats + labels]

# %% feature correlation with stress

# arrays to save correlation results
wellby_stress_correlations = []
wellby_sleep_correlations = []
feature_names = []

print("Point Biserial Correlation Analysis:")
print("====================================")

# correlation analysis for stress label
for feature in feats:
    stress_corr, p_value = pointbiserialr(df2[feature], df2["stress_binary"])
    sleep_corr, p_value = pointbiserialr(df2[feature], df2["sleep_binary"])
    wellby_stress_correlations.append(stress_corr)
    wellby_sleep_correlations.append(sleep_corr)
    feature_names.append(feature)

    print(f"{feature}: Correlation = {stress_corr:.2f}, p-value = {p_value:.3f}")
    print(f"{feature}: Correlation = {sleep_corr:.2f}, p-value = {p_value:.3f}")

#%% Create correlation heatmaps for Wellby dataset


wellby_correlation_data = pd.DataFrame({
    'Stress Correlation': wellby_stress_correlations,
    'Sleep Correlation': wellby_sleep_correlations
}, index=feature_names)

# Plot heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(wellby_correlation_data, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=0.5)

plt.title('HRV Features Correlation Comparison\nWellby Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('HRV Features', fontsize=12)
plt.xlabel('Dataset & Classification Type', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#%% plot all data together

all_correlation_data = pd.DataFrame({
    'Wellby Stress Correlation': wellby_stress_correlations,
    'Wellby Sleep Correlation': wellby_sleep_correlations,
    'WESAD Stress Correlation': wesad_correlations
}, index=feature_names)

plt.figure(figsize=(8, 10))
sns.heatmap(all_correlation_data, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=0.5)

plt.title('HRV Features Correlation Comparison\nWellby vs WESAD Datasets', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('HRV Features', fontsize=12)
plt.xlabel('Dataset & Classification Type', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
