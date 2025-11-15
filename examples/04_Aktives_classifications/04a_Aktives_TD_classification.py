"""
AKTIVES Dataset Stress Classification Evaluation
Outlined in Figure 5 and Table VI of the paper
10.1109/TAFFC.2025.3628467

Evaluates stress detection performance on the AKTIVES dataset using time-domain
HRV features with multiple random 70/30 train/test splits. Implements grid search
for hyperparameter tuning and calculates 95% confidence intervals across splits.

The AKTIVES dataset contains PPG data from children during therapeutic gaming
activities with expert-labeled stress annotations.

Dataset Reference:
B. Coşkun et al., "A physiological signal database of children with different 
special needs for stress recognition," Scientific Data, vol. 10, no. 1, p. 382, 2023.

Usage:
    - Update paths in configuration section below
    - Run entire script: python aktives_evaluation.py
    - Run interactively: Open in VSCode/Jupyter and execute cells with #%%

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%%
# Configuration
# TODO: Update paths based on local setup
DATA_PATH = '../../data/Aktives/extracted_features/ppg_features_combined.csv'
RESULTS_PATH = 'aktives_multiple_splits_with_ci.csv'

# Feature selection - exclude pNN50/pNN20 due to short window length in AKTIVES
FEATURES = ['HR_mean', 'HR_std', 'SDNN', 'SDSD', 'meanNN', 'medianNN', 'RMSSD', 'meanSD', 'sqi']
LABEL_COLUMN = 'label'

# Cross-validation configuration
N_SPLITS = 10  # Number of random 70/30 splits for evaluation
INNER_CV_FOLDS = 3  # Folds for hyperparameter tuning
TEST_SIZE = 0.3
RANDOM_STATE = 42

#%%
def calculate_confidence_intervals(values, confidence=0.95):
    """
    Calculate confidence intervals using t-distribution.
    
    Appropriate for small sample sizes (n=10 splits).
    
    Parameters:
    -----------
    values : array-like
        Performance scores across splits
    confidence : float, default=0.95
        Confidence level for interval
        
    Returns:
    --------
    tuple
        (mean, std, ci_lower, ci_upper)
    """
    if len(values) < 2:
        return np.mean(values), np.std(values), np.mean(values), np.mean(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * (std / np.sqrt(n))
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, std, ci_lower, ci_upper

#%%
# Load and prepare data
print("="*70)
print("AKTIVES DATASET STRESS CLASSIFICATION")
print("="*70)

data = pd.read_csv(DATA_PATH)

print(f"\nDataset Overview:")
print(f"  Shape: {data.shape}")
print(f"  Unique participants: {data['participant'].nunique()}")
print(f"  Label distribution:\n{data['label'].value_counts()}")
print(f"  Samples per participant:\n{data['participant'].value_counts().describe()}")

# One-hot encode cohort as contextual feature
cohort_dummies = pd.get_dummies(data['cohort'], prefix='cohort', drop_first=False)

X = pd.concat([data[FEATURES], cohort_dummies], axis=1).values
y = data[LABEL_COLUMN].values
groups = data['participant'].values

print(f"\nFeature Configuration:")
print(f"  Total features: {X.shape[1]} ({len(FEATURES)} HRV + {len(cohort_dummies.columns)} cohort)")

# Standardize features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#%%
# Define hyperparameter grids for grid search
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'bootstrap': [True, False]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    "K-Neighbors": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    },
    "LDA": {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
    },
    "SVM": {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'coef0': [0.0, 0.1, 0.5, 1.0]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
}

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

#%%
def run_multiple_splits_evaluation():
    """
    Run evaluation with multiple random 70/30 splits.
    
    For each split:
    1. Create stratified train/test split
    2. Tune hyperparameters using inner CV on training set
    3. Evaluate on held-out test set
    4. Record AUC-ROC, accuracy, and F1-score
    
    Calculates mean ± std and 95% CIs across all splits.
    
    Returns:
    --------
    pd.DataFrame
        Results with confidence intervals for each model
    """
    
    all_results = {model_name: {'auc_scores': [], 'accuracy_scores': [], 'f1_scores': []} 
                   for model_name in models.keys()}

    print(f"\n{'='*70}")
    print(f"RUNNING {N_SPLITS} RANDOM 70/30 SPLITS WITH GRID SEARCH")
    print(f"{'='*70}\n")

    for split_i in range(N_SPLITS):
        print(f"=== Split {split_i+1}/{N_SPLITS} ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, 
            random_state=RANDOM_STATE + split_i, 
            stratify=y
        )

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        for model_name, model in models.items():
            print(f"Training {model_name}...", end=' ')
            
            try:
                inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=inner_cv,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                test_auc = roc_auc_score(y_test, y_pred_proba)
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                
                all_results[model_name]['auc_scores'].append(test_auc * 100)
                all_results[model_name]['accuracy_scores'].append(test_acc * 100)
                all_results[model_name]['f1_scores'].append(test_f1 * 100)
                
                print(f"AUC={test_auc*100:.2f}, Acc={test_acc*100:.2f}, F1={test_f1*100:.2f}")
                
            except Exception as e:
                print(f"ERROR: {e}")
        
        print()

    print(f"{'='*70}")
    print("RESULTS ACROSS ALL SPLITS WITH 95% CONFIDENCE INTERVALS")
    print(f"{'='*70}\n")

    final_results = []

    for model_name in models.keys():
        if len(all_results[model_name]['auc_scores']) > 0:
            auc_mean, auc_std, auc_ci_low, auc_ci_high = calculate_confidence_intervals(
                all_results[model_name]['auc_scores'])
            acc_mean, acc_std, acc_ci_low, acc_ci_high = calculate_confidence_intervals(
                all_results[model_name]['accuracy_scores'])
            f1_mean, f1_std, f1_ci_low, f1_ci_high = calculate_confidence_intervals(
                all_results[model_name]['f1_scores'])
            
            final_results.append({
                'Algorithm': model_name,
                'AUC_Mean': auc_mean,
                'AUC_Std': auc_std,
                'AUC_CI_Lower': auc_ci_low,
                'AUC_CI_Upper': auc_ci_high,
                'AUC_Formatted': f"{auc_mean:.2f} ± {auc_std:.2f} ({auc_ci_low:.2f}, {auc_ci_high:.2f})",
                'Accuracy_Mean': acc_mean,
                'Accuracy_Std': acc_std,
                'Accuracy_CI_Lower': acc_ci_low,
                'Accuracy_CI_Upper': acc_ci_high,
                'Accuracy_Formatted': f"{acc_mean:.2f} ± {acc_std:.2f} ({acc_ci_low:.2f}, {acc_ci_high:.2f})",
                'F1_Mean': f1_mean,
                'F1_Std': f1_std,
                'F1_CI_Lower': f1_ci_low,
                'F1_CI_Upper': f1_ci_high,
                'F1_Formatted': f"{f1_mean:.2f} ± {f1_std:.2f} ({f1_ci_low:.2f}, {f1_ci_high:.2f})"
            })
            
            print(f"{model_name}:")
            print(f"  AUC:      {auc_mean:.2f} ± {auc_std:.2f} (95% CI: {auc_ci_low:.2f}, {auc_ci_high:.2f})")
            print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f} (95% CI: {acc_ci_low:.2f}, {acc_ci_high:.2f})")
            print(f"  F1 Score: {f1_mean:.2f} ± {f1_std:.2f} (95% CI: {f1_ci_low:.2f}, {f1_ci_high:.2f})")
            print()

    results_df = pd.DataFrame(final_results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Results saved to '{RESULTS_PATH}'")

    best_auc_idx = results_df['AUC_Mean'].idxmax()
    best_model = results_df.loc[best_auc_idx]
    print(f"\nBest performing model (by AUC): {best_model['Algorithm']}")
    print(f"AUC: {best_model['AUC_Formatted']}")
    
    return results_df

#%%
def display_summary(results_df):
    """
    Display formatted summary of results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from run_multiple_splits_evaluation
    """
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Model':<20} {'AUC-ROC':<35} {'Accuracy':<35}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['Algorithm']:<20} {row['AUC_Formatted']:<35} {row['Accuracy_Formatted']:<35}")

    print(f"\n{'='*70}")
    print("MODELS RANKED BY AUC-ROC")
    print(f"{'='*70}\n")
    
    results_sorted = results_df.sort_values('AUC_Mean', ascending=False)
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        print(f"{i+1}. {row['Algorithm']}: {row['AUC_Formatted']}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    results_df = run_multiple_splits_evaluation()
    display_summary(results_df)