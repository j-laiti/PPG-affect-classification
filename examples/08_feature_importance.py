"""
SHAP Feature Importance Analysis for Wellby Dataset
Figures 5 and 6 in the paper
DOI: 10.1109/TAFFC.2025.3628467

Generates SHAP (SHapley Additive exPlanations) visualizations to interpret
feature importance in stress and fatigue classification models.

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

#%%
# Configuration
# TODO: Update paths based on local setup
DATA_PATH = '../data/Wellby/Wellby_all_subjects_features.csv'
OUTPUT_DIR = '../results/Wellby/feature_importance/'

TASK = 'sleep'  # Options: 'stress' or 'sleep' (fatigue)
INCLUDE_DEMOGRAPHICS = False  # Set to True to include school, race, gender

# Matplotlib styling
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18

# Cross-validation configuration
N_SPLITS = 3
N_JOBS = 2
RANDOM_STATE = 0

#%%
# Feature definitions
BASE_FEATURES = {
    'stress': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 
               'pNN20', 'pNN50', 'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI'],
    'sleep': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD',
              'pNN20', 'pNN50', 'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI']
}

# Human-readable feature names for visualization
FEATURE_LABELS = {
    "HR_mean": "Mean Heart Rate",
    "HR_std": "Heart Rate Std",
    "meanNN": "Mean NN Interval",
    "SDNN": "SDNN",
    "medianNN": "Median NN Interval",
    "RMSSD": "RMSSD",
    "pNN20": "pNN20",
    "pNN50": "pNN50",
    "meanSD": "Mean of Successive Differences",
    "SDSD": "SDSD",
    "SQI": "Signal Quality Index",
    "PSS": "Perceived Stress Scale",
    "PSQI": "Pittsburgh Sleep Quality Index",
    "EPOCH": "Adolescent Wellbeing Questionnaire",
    # Demographics
    "School_Youthreach Rush": "School 3",
    "School_St. Wolstan's": "School 1",
    "School_Gorey Community School": "School 2",
    "Race/ethnicity_Black or Black Irish;": "Black or Black Irish",
    "Race/ethnicity_White - Other European background;": "White - Other European",
    "Race/ethnicity_White African;": "White African",
    "Race/ethnicity_White Irish;": "White Irish",
    "Race/ethnicity_Asian;": "Asian",
    "Gender_Female": "Female",
    "Gender_Male": "Male",
    "Gender_Prefer not to say": "Gender: Prefer not to say"
}

label = f'{TASK}_binary'
output_filename = f'feature_importance_{TASK}.png'

print("="*70)
print(f"SHAP FEATURE IMPORTANCE ANALYSIS - {TASK.upper()}")
print("="*70)
print(f"\nConfiguration:")
print(f"  Task: {TASK}")
print(f"  Include demographics: {INCLUDE_DEMOGRAPHICS}")
print(f"  Output file: {output_filename}")
print("-" * 70)

#%%
def prepare_features(data, task, include_demographics=False):
    """
    Prepare feature set with optional demographic variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw Wellby dataset
    task : str
        'stress' or 'sleep'
    include_demographics : bool
        Whether to one-hot encode and include school, race, gender
        
    Returns:
    --------
    tuple
        - feats: List of feature column names
        - formatted_feats: List of human-readable feature names
        - data: DataFrame with one-hot encoded demographics (if applicable)
    """
    feats = BASE_FEATURES[task].copy()
    
    if include_demographics:
        data = pd.get_dummies(data, columns=['School'], prefix='School')
        feats.extend([col for col in data.columns if col.startswith('School')])
        
        data = pd.get_dummies(data, columns=['Race/ethnicity'], prefix='Race/ethnicity')
        feats.extend([col for col in data.columns if col.startswith('Race/ethnicity')])
        
        data = pd.get_dummies(data, columns=['Gender'], prefix='Gender')
        feats.extend([col for col in data.columns if col.startswith('Gender')])
    
    # Ensure all features have labels (use original name if not in mapping)
    for feat in feats:
        if feat not in FEATURE_LABELS:
            FEATURE_LABELS[feat] = feat
    
    formatted_feats = [FEATURE_LABELS[feat] for feat in feats]
    
    print(f"\nFeature set prepared:")
    print(f"  Total features: {len(feats)}")
    print(f"  Features: {feats}")
    
    return feats, formatted_feats, data

#%%
# Load and prepare data
data = pd.read_csv(DATA_PATH)
feats, formatted_feats, data = prepare_features(data, TASK, INCLUDE_DEMOGRAPHICS)

X = data[feats].values
y = data[label].values
groups = data['Participant'].values

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Initialize cross-validation
kf = StratifiedGroupKFold(n_splits=N_SPLITS)

print(f"\nData loaded:")
print(f"  Total samples: {len(X)}")
print(f"  Unique participants: {len(np.unique(groups))}")
print(f"  Label distribution: {np.bincount(y.astype(int))}")

#%%
# Define hyperparameter grid for linear SVM
# Linear kernel required for SHAP interpretability
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear']  # Linear kernel for interpretability
}

model = svm.SVC(probability=True, class_weight='balanced')

#%%
def train_and_analyze_with_shap():
    """
    Train SVM model with grid search and compute SHAP feature importance.
    
    Process:
    1. Perform grid search with stratified group k-fold CV
    2. Train best model on full dataset
    3. Compute SHAP values for all samples
    4. Generate heatmap visualization
    5. Save feature importance rankings
    
    Returns:
    --------
    tuple
        - best_model: Trained SVM model
        - shap_values: SHAP explanation object
        - shap_importance: DataFrame with feature rankings
    """
    
    print(f"\n{'='*70}")
    print("TRAINING SVM WITH GRID SEARCH")
    print(f"{'='*70}\n")
    
    print("Performing grid search...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid_svm,
        cv=kf,
        scoring='average_precision',
        n_jobs=N_JOBS
    )
    grid_search.fit(X, y, groups=groups)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    
    # Evaluate model
    auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='roc_auc') * 100
    balanced_acc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                         scoring='balanced_accuracy') * 100
    auprc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                   scoring='average_precision') * 100
    
    print(f"\nModel Performance:")
    print(f"  AUPRC: {np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
    print(f"  AUC-ROC: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
    print(f"  Balanced Accuracy: {np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
    
    print(f"\n{'='*70}")
    print("COMPUTING SHAP VALUES")
    print(f"{'='*70}\n")
    
    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(best_model, X)
    
    print("Computing SHAP values for all samples...")
    shap_values = explainer(X)
    shap_values.feature_names = formatted_feats
    
    # Compute feature importance
    shap_importance = pd.DataFrame({
        "Feature": feats,
        "Formatted_Name": formatted_feats,
        "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(shap_importance[['Formatted_Name', 'Mean_Abs_SHAP']].head(10).to_string(index=False))
    
    return best_model, shap_values, shap_importance

#%%
def create_shap_heatmap(shap_values, task, output_path):
    """
    Create and save SHAP heatmap visualization.
    
    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values for all samples
    task : str
        'stress' or 'sleep' for plot title
    output_path : str
        Path to save figure
    """
    
    print(f"\n{'='*70}")
    print("CREATING SHAP HEATMAP")
    print(f"{'='*70}\n")
    
    task_name = "Stress" if task == "stress" else "Fatigue"
    
    fig = plt.figure(figsize=(12, 8))
    shap.plots.heatmap(shap_values, max_display=6, show=False)
    plt.title(f"{task_name} Classification Shapley Values", fontsize=20, pad=20)
    plt.xlabel("Recording Session", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"SHAP heatmap saved to: {output_path}")
    
    plt.show()

#%%
def save_feature_importance(shap_importance, task):
    """
    Save feature importance rankings to CSV.
    
    Parameters:
    -----------
    shap_importance : pd.DataFrame
        Feature importance rankings
    task : str
        'stress' or 'sleep'
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, f'{task}_shap_importance.csv')
    shap_importance.to_csv(output_path, index=False)
    print(f"Feature importance saved to: {output_path}")

#%%
if __name__ == "__main__":
    # Train model and compute SHAP values
    best_model, shap_values, shap_importance = train_and_analyze_with_shap()
    
    # Create visualization
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    create_shap_heatmap(shap_values, TASK, output_path)
    
    # Save feature importance
    save_feature_importance(shap_importance, TASK)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")