"""
HRV Feature Correlation Analysis

This script computes point-biserial correlations between time-domain HRV features
and binary stress/fatigue labels across three datasets (WESAD, AKTIVES, Wellby).
Generates Figure 4 from the manuscript.

Note: The Wellby dataset is not publicly available due to privacy restrictions.

Usage:
    - Run entire script: python correlation_analysis.py
    - Run interactively: Open in VSCode/Jupyter and execute cells with #%%
    - this depends on having the WESAD and AKTIVES features already extracted using
      the pipelines in src/feature_extraction/wesad_processing.py and
      src/feature_extraction/aktives_process_folder.py respectively.
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports and Configuration
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import numpy as np

# Time-domain HRV features to analyze
HRV_FEATURES = ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 
                'meanSD', 'SDSD', 'RMSSD', 'pNN20', 'pNN50']

# Feature display names for publication
FEATURE_LABELS = {
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

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman'],
    'mathtext.fontset': 'stix'
})


#%% Helper Functions
def compute_correlations(df, features, label_column):
    """
    Compute point-biserial correlations between features and binary label.
    
    Parameters
    ----------
    df : DataFrame
        Data containing features and labels
    features : list
        List of feature column names
    label_column : str
        Name of binary label column
        
    Returns
    -------
    correlations : list
        Correlation coefficients for each feature
    p_values : list
        P-values for each correlation
    """
    correlations = []
    p_values = []
    
    print(f"\n=== {label_column.upper()} CORRELATION ANALYSIS ===")
    print(f"Label distribution: {df[label_column].value_counts().to_dict()}")
    print("\nCorrelations:")
    print("-" * 50)
    
    for feature in features:
        corr, p_val = pointbiserialr(df[feature], df[label_column])
        correlations.append(corr)
        p_values.append(p_val)
        
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feature:12s}: r = {corr:6.3f}, p = {p_val:.4f} {sig_marker}")
    
    return correlations, p_values


def plot_correlation_heatmap(correlation_df, title, figsize=(8, 10), 
                             save_path=None):
    """
    Create correlation heatmap visualization.
    
    Parameters
    ----------
    correlation_df : DataFrame
        DataFrame with features as index and datasets/labels as columns
    title : str
        Plot title
    figsize : tuple
        Figure dimensions (width, height)
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(correlation_df, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                vmin=-1.0,
                vmax=1.0,
                cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=0.5,
                annot_kws={'fontsize': 14},
                ax=ax)
    
    # Customize colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation Coefficient', fontsize=18)
    
    # Set labels and title
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_ylabel('HRV Features', fontsize=20)
    ax.set_xlabel('Dataset & Classification Type', fontsize=20)
    ax.tick_params(axis='x', labelsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


#%% WESAD Dataset Analysis
print("\n" + "="*60)
print("WESAD DATASET")
print("="*60)

wesad_df = pd.read_csv("../compiled_features_data/WESAD_features_merged_bp_time.csv") # features extracted using the pipeline in src/feature_extraction/wesad_processing.py 
wesad_corr, wesad_pvals = compute_correlations(
    wesad_df, 
    HRV_FEATURES, 
    'label'
)


#%% AKTIVES Dataset Analysis
print("\n" + "="*60)
print("AKTIVES DATASET")
print("="*60)

aktives_df = pd.read_csv("../data/Aktives/extracted_features/ppg_features_combined.csv") # features extracted using the pipeline in src/feature_extraction/aktives_process_folder.py  
aktives_corr, aktives_pvals = compute_correlations(
    aktives_df,
    HRV_FEATURES,
    'label'
)


#%% Wellby Dataset Analysis
print("\n" + "="*60)
print("WELLBY DATASET")
print("="*60)

wellby_df = pd.read_csv("../data/Wellby/Wellby_all_subjects_features.csv")

# Stress correlations
wellby_stress_corr, wellby_stress_pvals = compute_correlations(
    wellby_df,
    HRV_FEATURES,
    'stress_binary'
)

# Fatigue correlations
wellby_fatigue_corr, wellby_fatigue_pvals = compute_correlations(
    wellby_df,
    HRV_FEATURES,
    'sleep_binary'
)


#%% Individual Dataset Heatmaps (Optional)
# Uncomment to generate individual heatmaps for each dataset

# WESAD only
# wesad_df_plot = pd.DataFrame({
#     'Stress': wesad_corr
# }, index=[FEATURE_LABELS[f] for f in HRV_FEATURES])
# plot_correlation_heatmap(wesad_df_plot, 'WESAD Dataset Correlations', 
#                          figsize=(4, 8))

# AKTIVES only
# aktives_df_plot = pd.DataFrame({
#     'Stress': aktives_corr
# }, index=[FEATURE_LABELS[f] for f in HRV_FEATURES])
# plot_correlation_heatmap(aktives_df_plot, 'AKTIVES Dataset Correlations',
#                          figsize=(4, 8))

# Wellby only
# wellby_df_plot = pd.DataFrame({
#     'Stress': wellby_stress_corr,
#     'Fatigue': wellby_fatigue_corr
# }, index=[FEATURE_LABELS[f] for f in HRV_FEATURES])
# plot_correlation_heatmap(wellby_df_plot, 'Wellby Dataset Correlations',
#                          figsize=(6, 8))


#%% Combined Correlation Heatmap (Figure 4)
# Combine all correlations into single DataFrame for comparison

combined_correlations = pd.DataFrame({
    'Wellby Stress': wellby_stress_corr,
    'Wellby Fatigue': wellby_fatigue_corr,
    'AKTIVES Stress': aktives_corr,
    'WESAD Stress': wesad_corr
}, index=[FEATURE_LABELS[f] for f in HRV_FEATURES])

# Generate publication-ready figure
plot_correlation_heatmap(
    combined_correlations,
    title='Point-Biserial Correlation Across Datasets',
    figsize=(9, 8),
    save_path='../results/correlation_heatmap.png'
)
# %%
