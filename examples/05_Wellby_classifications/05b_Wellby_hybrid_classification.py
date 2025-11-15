"""
Wellby Hybrid Features (CNN+TD) Stress and Fatigue Classification

Evaluates stress and fatigue detection on the Wellby dataset using hybrid features
combining CNN-extracted deep features with traditional time-domain HRV features.
Uses stratified group k-fold cross-validation to prevent data leakage from the same
participant appearing in both train and test sets.

The hybrid features combine:
- 32 CNN-extracted features from raw PPG signals
- Time-domain HRV features (HR metrics, SDNN, RMSSD, etc.)
- Signal quality indices and baseline surveys (where applicable)

Note: The Wellby dataset is not publicly available due to privacy restrictions.
      Separate feature files exist for stress and fatigue classification tasks.

Dataset Reference:
J. Laiti et al., "Real-World Classification of Student Stress and Fatigue Using 
Wearable PPG Recordings," Journal of Affective Computing, 2025.

Usage:
    - Update TASK in configuration section below ('stress' or 'sleep')
    - Run entire script: python wellby_hybrid_evaluation.py
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
TASK = 'stress'  # Options: 'stress' or 'sleep' (fatigue)

# Dynamic file paths based on task
DATA_PATHS = {
    'stress': '../../data/Wellby_hybrid_features/wellby_stress_hybrid_features.csv',
    'sleep': '../../data/Wellby_hybrid_features/wellby_sleep_hybrid_features.csv'
}

RESULTS_DIR = '../../results/Wellby/hybrid_classification_test/'

# Cross-validation configuration
N_SPLITS = 3  # Stratified group k-fold splits
N_JOBS = 2  # Parallel jobs for grid search
RANDOM_STATE = 0

# Validate and set paths
if TASK not in DATA_PATHS:
    raise ValueError("TASK must be 'stress' or 'sleep'")

DATA_PATH = DATA_PATHS[TASK]
output_filename = f'Wellby_{TASK}_hybrid_eval.csv'

print("="*70)
print(f"WELLBY HYBRID FEATURES (CNN+TD) {TASK.upper()} CLASSIFICATION")
print("="*70)
print(f"\nConfiguration:")
print(f"  Task: {TASK}")
print(f"  Data path: {DATA_PATH}")
print(f"  Output file: {output_filename}")
print(f"  Cross-validation: {N_SPLITS}-fold stratified group k-fold")
print("-" * 70)

#%%
def prepare_hybrid_features(data):
    """
    Prepare hybrid feature set from CNN and time-domain features.
    
    Automatically identifies CNN features (prefixed with 'cnn_feature_'),
    time-domain HRV features, and separates metadata columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw hybrid features dataframe
        
    Returns:
    --------
    tuple
        - X: Standardized feature matrix
        - y: Binary labels
        - groups: Participant IDs for group k-fold
        - feature_info: Dictionary with feature composition details
    """
    cnn_features = [col for col in data.columns if col.startswith('cnn_feature_')]
    
    metadata_cols = ['participant', 'session_id', 'school', 'age', 'gender', 'label']
    td_features = [col for col in data.columns if col not in cnn_features and col not in metadata_cols]
    
    print(f"\nFeature Composition:")
    print(f"  CNN features: {len(cnn_features)}")
    print(f"  TD features: {len(td_features)}")
    print(f"  TD feature names: {td_features}")
    
    all_features = cnn_features + td_features
    
    X = data[all_features].values
    y = data['label'].values
    groups = data['participant'].values
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    feature_info = {
        'cnn_count': len(cnn_features),
        'td_count': len(td_features),
        'total_count': len(all_features)
    }
    
    print(f"  Total features: {feature_info['total_count']}")
    print(f"\nData loaded:")
    print(f"  Total samples: {len(X)}")
    print(f"  Unique participants: {len(np.unique(groups))}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")
    print(f"  Class balance: {np.bincount(y.astype(int))[1]/len(y)*100:.1f}% positive")
    
    return X, y, groups, feature_info

#%%
# Load and prepare hybrid features
data = pd.read_csv(DATA_PATH)
X, y, groups, feature_info = prepare_hybrid_features(data)

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
    Run grid search with stratified group k-fold cross-validation on hybrid features.
    
    For each model:
    1. Perform grid search to find best hyperparameters using AUPRC
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
    print("HYBRID FEATURES (CNN+TD) RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Feature Composition: {feature_info['cnn_count']} CNN + "
          f"{feature_info['td_count']} TD = {feature_info['total_count']} total features\n")
    
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    results_df = run_grid_search_evaluation()
    save_and_display_results(results_df)