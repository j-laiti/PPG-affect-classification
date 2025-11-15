"""
Demographic Bias Analysis for Wellby Stress/Fatigue Classification
Table XI in the paper
DOI: 10.1109/TAFFC.2025.3628467

Evaluates potential demographic bias in stress and fatigue classification models
by analyzing prediction accuracy across different demographic subgroups (gender,
race/ethnicity, school). Uses stratified group k-fold cross-validation to assess
model fairness and identify disparities in performance.

Author: Justin Laiti
Note: Code final organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

#%%
# Configuration
# TODO: Update paths based on local setup
DATA_PATH = 'data/features/Wellby_all_subjects_features.csv'
OUTPUT_DIR = 'results/Wellby/demographic_bias/'

TASK = 'sleep'  # Options: 'stress' or 'sleep' (fatigue)

# Feature set - all time-domain features plus contextual data
FEATURES = ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD', 
            'pNN20', 'pNN50', 'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI']

# Cross-validation configuration
N_SPLITS = 3
N_JOBS = 2
RANDOM_STATE = 0

label = f'{TASK}_binary'

print("="*70)
print(f"DEMOGRAPHIC BIAS ANALYSIS - {TASK.upper()} CLASSIFICATION")
print("="*70)
print(f"\nConfiguration:")
print(f"  Task: {TASK}")
print(f"  Features: {len(FEATURES)} time-domain + contextual")
print(f"  Label: {label}")
print(f"  Cross-validation: {N_SPLITS}-fold stratified group k-fold")
print("-" * 70)

#%%
def load_and_prepare_data(data_path, features, label_col):
    """
    Load Wellby dataset and extract demographic information.
    
    Preserves demographic labels (school, race/ethnicity, gender) before
    standardization for later stratified analysis of model performance.
    
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
        - demographics: Dict with school, race, gender labels
    """
    data = pd.read_csv(data_path)
    
    # Extract demographic information before processing
    demographics = {
        'school': data['School'].copy().values,
        'race': data['Race/ethnicity'].copy().values,
        'gender': data['Gender'].copy().values
    }
    
    X = data[features].values
    y = data[label_col].values
    groups = data['Participant'].values
    
    # Standardize features
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    print(f"\nData loaded:")
    print(f"  Total samples: {len(X)}")
    print(f"  Unique participants: {len(np.unique(groups))}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")
    print(f"\nDemographic distribution:")
    print(f"  Gender: {np.unique(demographics['gender'], return_counts=True)}")
    print(f"  Schools: {np.unique(demographics['school'], return_counts=True)}")
    print(f"  Unique races/ethnicities: {len(np.unique(demographics['race']))}")
    
    return X, y, groups, demographics

#%%
# Load and prepare data
X, y, groups, demographics = load_and_prepare_data(DATA_PATH, FEATURES, label)

# Initialize stratified group k-fold cross-validation
kf = StratifiedGroupKFold(n_splits=N_SPLITS)

#%%
# Hyperparameter grid for SVM
# SVM chosen based on strong performance in main evaluation
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

model = svm.SVC(probability=True, class_weight='balanced')

#%%
def train_and_evaluate_with_demographics():
    """
    Train SVM with grid search and log predictions with demographic info.
    
    For each fold in cross-validation:
    1. Train model on training set
    2. Make predictions on test set
    3. Log each prediction with:
       - True label and predicted label
       - Whether prediction was correct
       - Demographic information (gender, race, school)
    
    This enables stratified analysis of model performance across subgroups.
    
    Returns:
    --------
    tuple
        - results_df: Overall model performance
        - prediction_df: Per-sample predictions with demographics
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
    
    # Evaluate overall performance
    auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='roc_auc') * 100
    auprc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                   scoring='average_precision') * 100
    
    print(f"\nOverall Model Performance:")
    print(f"  AUC-ROC: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
    print(f"  AUPRC: {np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
    
    results = {
        "Model": ["SVM"],
        "AUC-ROC": [f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}"],
        "Average Precision (AUPRC)": [f"{np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}"]
    }
    
    # Log per-sample predictions with demographic information
    print(f"\n{'='*70}")
    print("LOGGING PREDICTIONS BY DEMOGRAPHIC GROUPS")
    print(f"{'='*70}\n")
    
    prediction_log = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        for i, idx in enumerate(test_idx):
            prediction_log.append({
                "Fold": fold,
                "SampleIndex": idx,
                "TrueLabel": y_test[i],
                "PredictedLabel": y_pred[i],
                "Correct": int(y_test[i] == y_pred[i]),
                "Gender": demographics['gender'][idx],
                "Race/ethnicity": demographics['race'][idx],
                "School": demographics['school'][idx]
            })
    
    prediction_df = pd.DataFrame(prediction_log)
    print(f"Logged {len(prediction_df)} predictions across {N_SPLITS} folds")
    
    return pd.DataFrame(results), prediction_df

#%%
def categorize_race(race):
    """
    Group race/ethnicity into White/Non-White categories.
    
    Simplifies race analysis by creating binary grouping while
    preserving detailed breakdown in raw predictions.
    
    Parameters:
    -----------
    race : str
        Original race/ethnicity label
        
    Returns:
    --------
    str
        'White', 'Non-White', or 'Unknown'
    """
    if pd.isna(race):
        return 'Unknown'
    elif 'White' in str(race):
        return 'White'
    else:
        return 'Non-White'

def analyze_demographic_bias(prediction_df):
    """
    Analyze model performance across demographic subgroups.
    
    Calculates accuracy (percent correct predictions) stratified by:
    - Gender (male, female, prefer not to say)
    - Race/ethnicity (detailed categories)
    - Race/ethnicity (grouped: White vs. Non-White)
    - School (School 1, School 2, School 3)
    
    Parameters:
    -----------
    prediction_df : pd.DataFrame
        Per-sample predictions with demographic labels
    """
    
    print(f"\n{'='*70}")
    print("DEMOGRAPHIC BIAS ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    
    # Gender analysis
    print("Accuracy by Gender:")
    print("-" * 40)
    gender_summary = prediction_df.groupby("Gender")["Correct"].agg(['mean', 'count'])
    gender_summary['mean'] = (gender_summary['mean'] * 100).round(2)
    gender_summary.columns = ['Accuracy (%)', 'Sample Size']
    print(gender_summary)
    
    # Detailed race/ethnicity analysis
    print("\nAccuracy by Race/Ethnicity (detailed):")
    print("-" * 40)
    race_summary = prediction_df.groupby("Race/ethnicity")["Correct"].agg(['mean', 'count'])
    race_summary['mean'] = (race_summary['mean'] * 100).round(2)
    race_summary.columns = ['Accuracy (%)', 'Sample Size']
    print(race_summary)
    
    # Grouped race analysis (White vs. Non-White)
    prediction_df['Race_grouped'] = prediction_df['Race/ethnicity'].apply(categorize_race)
    
    print("\nAccuracy by Race/Ethnicity (grouped):")
    print("-" * 40)
    race_grouped_summary = prediction_df.groupby("Race_grouped")["Correct"].agg(['mean', 'count'])
    race_grouped_summary['mean'] = (race_grouped_summary['mean'] * 100).round(2)
    race_grouped_summary.columns = ['Accuracy (%)', 'Sample Size']
    print(race_grouped_summary)
    
    # School analysis
    print("\nAccuracy by School:")
    print("-" * 40)
    school_summary = prediction_df.groupby("School")["Correct"].agg(['mean', 'count'])
    school_summary['mean'] = (school_summary['mean'] * 100).round(2)
    school_summary.columns = ['Accuracy (%)', 'Sample Size']
    print(school_summary)
    
    # Identify largest disparities
    print(f"\n{'='*70}")
    print("DISPARITY SUMMARY")
    print(f"{'='*70}\n")
    
    gender_disparity = gender_summary['Accuracy (%)'].max() - gender_summary['Accuracy (%)'].min()
    race_disparity = race_grouped_summary['Accuracy (%)'].max() - race_grouped_summary['Accuracy (%)'].min()
    school_disparity = school_summary['Accuracy (%)'].max() - school_summary['Accuracy (%)'].min()
    
    print(f"Maximum accuracy disparity:")
    print(f"  Gender: {gender_disparity:.2f} percentage points")
    print(f"  Race (grouped): {race_disparity:.2f} percentage points")
    print(f"  School: {school_disparity:.2f} percentage points")

def save_results(results_df, prediction_df, task):
    """
    Save results to CSV files.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Overall model performance
    prediction_df : pd.DataFrame
        Per-sample predictions with demographics
    task : str
        'stress' or 'sleep'
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_path = os.path.join(OUTPUT_DIR, f'{task}_overall_performance.csv')
    predictions_path = os.path.join(OUTPUT_DIR, f'{task}_predictions_by_demographics.csv')
    
    results_df.to_csv(results_path, index=False)
    prediction_df.to_csv(predictions_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  {results_path}")
    print(f"  {predictions_path}")

#%%
if __name__ == "__main__":
    # Train model and log predictions
    results_df, prediction_df = train_and_evaluate_with_demographics()
    
    print("\nOverall Performance:")
    print(results_df.to_string(index=False))
    
    # Analyze demographic bias
    analyze_demographic_bias(prediction_df)
    
    # Save results
    save_results(results_df, prediction_df, TASK)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")