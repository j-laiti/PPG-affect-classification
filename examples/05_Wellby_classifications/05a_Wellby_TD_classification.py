"""
Wellby Dataset Stress and Fatigue Classification Evaluation
Tables VII and VIII in the paper
10.1109/TAFFC.2025.3628467

Evaluates stress and fatigue detection performance on the real-world Wellby dataset
collected from secondary school students. Uses stratified group k-fold cross-validation
with grid search hyperparameter tuning to evaluate multiple feature sets.

The Wellby dataset contains PPG recordings from students during their daily lives,
with self-reported stress and fatigue labels. Features include time-domain HRV metrics,
signal quality indices, and baseline well-being questionnaires (PSS, PSQI, EPOCH).

Note: The Wellby dataset is not publicly available due to privacy restrictions.

Dataset Reference:
J. Laiti et al., "Real-World Classification of Student Stress and Fatigue Using 
Wearable PPG Recordings," Journal of Affective Computing, 2025.

Usage:
    - Update TASK and FEATURE_SET in configuration section below
    - Run entire script: python wellby_evaluation.py
    - Run interactively: Open in VSCode/Jupyter and execute cells with #%%

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
import os

#%%
# Configuration
# TODO: Update paths based on local setup
DATA_PATH = '../../data/Wellby/Wellby_all_subjects_features.csv'
RESULTS_DIR = '../../results/Wellby/TD_classification/'

# Task Configuration - Change these to evaluate different scenarios
TASK = 'stress'  # Options: 'stress' or 'sleep' (fatigue)
FEATURE_SET = 'td_best_surveys_sqi'  # See feature_sets below for options

# Cross-validation configuration
N_SPLITS = 3  # Stratified group k-fold splits
N_JOBS = 2  # Parallel jobs for grid search
RANDOM_STATE = 0

#%%
# Feature Set Definitions
# Different combinations of HRV features, signal quality, and baseline surveys
feature_sets = {
    'stress': {
        'td_best': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50'],
        'td_best_sqi': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50', 'SQI'],
        'td_best_surveys': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50', 
                           'PSS', 'PSQI', 'EPOCH'],
        'td_best_surveys_sqi': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50',
                               'PSS', 'PSQI', 'EPOCH', 'SQI'],
        'all_td': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50',
                  'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI']
    },
    'sleep': {
        'td_best': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'meanSD', 'pNN20', 'pNN50'],
        'td_best_sqi': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'meanSD', 'pNN20', 'pNN50', 'SQI'],
        'td_best_surveys': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'meanSD', 'pNN20', 'pNN50',
                           'PSS', 'PSQI', 'EPOCH'],
        'td_best_surveys_sqi': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'meanSD', 'pNN20', 'pNN50',
                               'PSS', 'PSQI', 'EPOCH', 'SQI'],
        'all_td': ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 'pNN20', 'pNN50',
                  'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI']
    }
}

# Generate configuration based on task selection
label = f'{TASK}_binary'
feats = feature_sets[TASK][FEATURE_SET]
output_filename = f'Wellby_{TASK}_{FEATURE_SET}_eval.csv'

print("="*70)
print(f"WELLBY {TASK.upper()} CLASSIFICATION EVALUATION")
print("="*70)
print(f"\nConfiguration:")
print(f"  Task: {TASK}")
print(f"  Feature Set: {FEATURE_SET}")
print(f"  Number of features: {len(feats)}")
print(f"  Features: {feats}")
print(f"  Label column: {label}")
print(f"  Output file: {output_filename}")
print(f"  Cross-validation: {N_SPLITS}-fold stratified group k-fold")
print("-" * 70)

#%%
def load_and_prepare_data(data_path, features, label_col):
    """
    Load Wellby dataset and prepare features for classification.
    
    Uses participant IDs as groups to ensure samples from the same student
    are kept together during cross-validation splits.
    
    Parameters:
    -----------
    data_path : str
        Path to Wellby features CSV
    features : list
        List of feature column names
    label_col : str
        Name of label column (stress_binary or sleep_binary)
        
    Returns:
    --------
    tuple
        - X: Standardized feature matrix
        - y: Binary labels
        - groups: Participant IDs for group k-fold
    """
    data = pd.read_csv(data_path)
    
    X = data[features].values
    y = data[label_col].values
    groups = data['Participant'].values
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    print(f"\nData loaded:")
    print(f"  Total samples: {len(X)}")
    print(f"  Unique participants: {len(np.unique(groups))}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")
    print(f"  Class balance: {np.bincount(y.astype(int))[1]/len(y)*100:.1f}% positive")
    
    return X, y, groups

#%%
# Load and prepare data
X, y, groups = load_and_prepare_data(DATA_PATH, feats, label)

# Initialize stratified group k-fold cross-validation
kf = StratifiedGroupKFold(n_splits=N_SPLITS)

#%%
# Define hyperparameter grids for each model
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    "AdaBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    "K-Neighbors": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    "LDA": {
        'solver': ['lsqr'],
        'shrinkage': [None, 'auto']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['linear', 'rbf']
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Initialize models with appropriate configurations
models = {
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

#%%
def run_grid_search_evaluation():
    """
    Run grid search with stratified group k-fold cross-validation.
    
    For each model:
    1. Perform grid search to find best hyperparameters
    2. Evaluate best model using cross-validation on three metrics:
       - Average Precision (AUPRC) - primary metric for imbalanced data
       - AUC-ROC
       - Balanced Accuracy
    
    Uses StratifiedGroupKFold to ensure:
    - Same participant's samples stay together in train or test
    - Class balance is maintained across folds
    
    Returns:
    --------
    pd.DataFrame
        Results with mean ± std for each metric and best parameters
    """
    
    results = {
        "Model": [],
        "Average Precision (AUPRC)": [],
        "AUC-ROC": [],
        "Balanced Accuracy": [],
        "Best Parameters": []
    }

    print(f"\n{'='*70}")
    print("RUNNING GRID SEARCH WITH STRATIFIED GROUP K-FOLD CV")
    print(f"{'='*70}\n")

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=kf,
                scoring='average_precision',
                n_jobs=N_JOBS
            )
            grid_search.fit(X, y, groups=groups)
            
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='roc_auc') * 100
            balanced_acc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                                 scoring='balanced_accuracy') * 100
            auprc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                          scoring='average_precision') * 100

            results["Model"].append(model_name)
            results["Average Precision (AUPRC)"].append(f"{np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
            results["AUC-ROC"].append(f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
            results["Balanced Accuracy"].append(f"{np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
            results["Best Parameters"].append(best_params)
            
            print(f"  AUPRC: {np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
            print(f"  AUC-ROC: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
            print(f"  Balanced Acc: {np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}\n")

    return pd.DataFrame(results)

#%%
def save_and_display_results(results_df):
    """
    Save results to CSV and display summary.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from grid search evaluation
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_filename)
    results_df.to_csv(output_path, index=False)
    
    print(f"{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    results_df = run_grid_search_evaluation()
    save_and_display_results(results_df)