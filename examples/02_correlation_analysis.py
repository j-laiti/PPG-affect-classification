# The purpose of this file is to create correlation graphs between HRV features and stress/fatigue labels
# Figure 4 in the manuscript was created using this code.
# Justin Laiti June 23 2025

### WESAD Correlation Analysis ###

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

print("=== WESAD LABEL ANALYSIS ===")
print("Label value counts:", df2['label'].value_counts())
print("Feature means by label:")
for label_val in df2['label'].unique():
    subset = df2[df2['label'] == label_val]
    print(f"\nLabel {label_val}:")
    print(f"  HR_mean: {subset['HR_mean'].mean():.2f}")
    print(f"  RMSSD: {subset['RMSSD'].mean():.2f}")

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




# %%
# ### AKTIVES Correlation Analysis ###

# Load both CSV files
aktives_features_file_path = "../data/Aktives/extracted_features/ppg_features_combined.csv"

# Load datasets & add a column indicating the processing pipeline
df = pd.read_csv(aktives_features_file_path)

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50']
label = ['label']

df2 = df[feats + label]

# %% feature correlation with stress

# arrays to save correlation results
aktives_correlations = []
feature_names = []

print("Point Biserial Correlation Analysis:")
print("====================================")

# correlation analysis for stress label
for feature in feats:
    stress_corr, p_value = pointbiserialr(df2[feature], df2["label"])
    aktives_correlations.append(stress_corr)
    feature_names.append(feature)

    print(f"{feature}: Correlation = {stress_corr:.2f}, p-value = {p_value:.3f}")

#%% Create correlation heatmaps for Aktives dataset
aktives_correlation_data = pd.DataFrame({
    'Stress Correlation': aktives_correlations,
}, index=feature_names)

# Plot heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(aktives_correlation_data, 
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
    'Wellby Stress': wellby_stress_correlations,
    'Wellby Sleep': wellby_sleep_correlations,
    'Aktives Stress': aktives_correlations,
    'WESAD Stress': wesad_correlations
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

# Set Times font for manuscript
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman'],
    'mathtext.fontset': 'stix'
})

# Create mapping for better feature names
feature_name_mapping = {
    'HR_mean': 'Mean HR',
    'HR_std': 'Std HR', 
    'meanNN': 'Mean NN',
    'SDNN': 'SDNN',
    'medianNN': 'Median NN',
    'meanSD': 'Mean SD',
    'SDSD': 'SDSD',
    'RMSSD': 'RMSSD',
    'pNN20': 'pNN20',
    'pNN50': 'pNN50'
}

# Apply mapping to feature names
renamed_features = [feature_name_mapping.get(name, name) for name in feature_names]

all_correlation_data = pd.DataFrame({
    'Wellby Stress': wellby_stress_correlations,
    'Wellby Sleep': wellby_sleep_correlations,
    'AKTIVES Stress': aktives_correlations,
    'WESAD Stress': wesad_correlations
}, index=renamed_features)

# Create manuscript-ready figure
plt.figure(figsize=(9, 8))
sns.heatmap(all_correlation_data, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            vmin=-1.0,
            vmax=1.0,     
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.9},
            linewidths=0.5,
            annot_kws={'fontsize': 14})  # Annotation font size

plt.title('Point-Biserial Correlation Across Datasets', 
          fontsize=20, fontweight='bold', pad=20, fontfamily='serif')
plt.ylabel('HRV Features', fontsize=18, fontfamily='serif')
plt.xlabel('Dataset & Classification Type', fontsize=18, fontfamily='serif')
plt.xticks(rotation=0, fontsize=16)
plt.yticks(rotation=0, fontsize=16)

plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Adjust layout for better spacing
plt.tight_layout()

# Optional: Save as high-quality figure for manuscript
# plt.savefig('hrv_correlations_manuscript.pdf', dpi=300, bbox_inches='tight', 
#             facecolor='white', edgecolor='none')
# plt.savefig('hrv_correlations_manuscript.png', dpi=300, bbox_inches='tight')

plt.show()

# now save it and comment on it!!!
# %%
plt.figure(figsize=(9, 8))
ax = sns.heatmap(all_correlation_data, 
                 annot=True, 
                 fmt='.2f', 
                 cmap='coolwarm',
                 center=0,
                 vmin=-1.0,
                 vmax=1.0,     
                 cbar_kws={'label': 'Correlation Coefficient'},
                 linewidths=0.5,
                 annot_kws={'fontsize': 14})

# Manually set the colorbar label font size
cbar = ax.collections[0].colorbar
cbar.set_label('Correlation Coefficient', fontsize=18)

plt.title('Point-Biserial Correlation Across Datasets', 
          fontsize=22, fontweight='bold', pad=20, fontfamily='serif')
plt.ylabel('HRV Features', fontsize=20, fontfamily='serif')
plt.xlabel('Dataset & Classification Type', fontsize=20, fontfamily='serif')
plt.xticks(rotation=45, fontsize=18)
plt.yticks(rotation=0, fontsize=18)

plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
# %%
