"""
Leave-One-Group-Out (LOGO) Cross-Validation for Wellby Dataset
Table IX in the paper
DOI: 10.1109/TAFFC.2025.3628467

Evaluates stress and fatigue classification using Leave-One-Group-Out cross-validation,
where each participant is held out as a test set while training on all other participants.

The Wellby dataset is not publicly available due to privacy restrictions.

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold, BaseCrossValidator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

#%%
# Configuration
# TODO: Update paths based on local setup
DATA_PATH = 'data/features/combined_sim_features_3inc.csv'
OUTPUT_DIR = 'results/Wellby/LOGO_validation/'
INTERIM_RESULTS_PATH = 'interim_model_evaluations.csv'
FINAL_RESULTS_PATH = 'all_model_outputs.csv'

TASK = 'stress'  # Options: 'stress' or 'sleep' (fatigue)

# Feature set - all time-domain features plus contextual data
FEATURES = ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'RMSSD',
            'pNN20', 'pNN50', 'meanSD', 'SDSD', 'PSS', 'PSQI', 'EPOCH', 'SQI']

# Cross-validation configuration
N_JOBS = 2
RANDOM_STATE = 0

label = f'{TASK}_binary'

print("="*70)
print(f"LOGO CROSS-VALIDATION - {TASK.upper()} CLASSIFICATION")
print("="*70)
print(f"\nConfiguration:")
print(f"  Task: {TASK}")
print(f"  Features: {len(FEATURES)} time-domain + contextual")
print(f"  Label: {label}")
print(f"  Validation: Leave-One-Group-Out (per participant)")
print("-" * 70)

#%%
class StratifiedLeaveOneGroupOut(BaseCrossValidator):
    """
    Custom LOGO cross-validator that only includes valid splits.
    
    A split is considered valid if both train and test sets contain
    samples from both classes. This prevents evaluation errors when
    a participant only has samples from one class.
    
    Attributes:
    -----------
    groups : array-like
        Participant IDs for each sample
    y : array-like
        Binary labels for each sample
    valid_splits : int
        Count of valid splits found
    """
    
    def __init__(self, groups, y):
        self.groups = groups
        self.y = y
        self.valid_splits = 0

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits, yielding only valid ones.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like, optional
            Labels (uses self.y if not provided)
        groups : array-like, optional
            Groups (uses self.groups if not provided)
            
        Yields:
        -------
        tuple
            (train_indices, test_indices) for each valid split
        """
        unique_groups = np.unique(self.groups)
        self.valid_splits = 0
        
        for test_group in unique_groups:
            train_idx = np.where(self.groups != test_group)[0]
            test_idx = np.where(self.groups == test_group)[0]
            
            # Only yield if both train and test have both classes
            if (len(np.unique(self.y[train_idx])) > 1 and 
                len(np.unique(self.y[test_idx])) > 1):
                self.valid_splits += 1
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Get number of valid splits.
        
        Iterates through all potential splits to count valid ones.
        
        Returns:
        --------
        int
            Number of valid LOGO splits
        """
        self.valid_splits = 0
        list(self.split(X, y, groups))
        return self.valid_splits

#%%
def load_and_prepare_data(data_path, features, label_col):
    """
    Load Wellby dataset and prepare for LOGO validation.
    
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
        - groups: Participant IDs
        - gender_labels: Gender for each sample
        - cv: Initialized LOGO cross-validator
    """
    data = pd.read_csv(data_path)
    
    X = data[features].values
    y = data[label_col].values
    groups = data['Participant'].values
    gender_labels = data['Gender'].values
    
    # Standardize features
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    # Initialize LOGO cross-validator
    cv = StratifiedLeaveOneGroupOut(groups=groups, y=y)
    n_splits = cv.get_n_splits(X, y, groups)
    
    print(f"\nData loaded:")
    print(f"  Total samples: {len(X)}")
    print(f"  Unique participants: {len(np.unique(groups))}")
    print(f"  Valid LOGO splits: {n_splits}")
    print(f"  Label distribution: {np.bincount(y.astype(int))}")
    
    if n_splits < len(np.unique(groups)):
        excluded = len(np.unique(groups)) - n_splits
        print(f"  Note: {excluded} participants excluded (single class only)")
    
    return X, y, groups, gender_labels, cv

#%%
# Load and prepare data
X, y, groups, gender_labels, kf = load_and_prepare_data(DATA_PATH, FEATURES, label)

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
def run_logo_validation_with_demographics():
    """
    Run LOGO validation with gender-stratified performance tracking.
    
    For each model:
    1. Perform grid search using LOGO cross-validation
    2. Evaluate overall performance (AUC-ROC, AUPRC, balanced accuracy)
    3. Track per-gender performance (accuracy, F1-score) across folds
    4. Save interim results to prevent data loss
    
    Warning: LOGO validation is computationally expensive and may have
    high variance due to small sample sizes per participant.
    
    Returns:
    --------
    tuple
        - results_df: Overall model performance
        - gender_df: Gender-stratified performance metrics
    """
    
    results = {
        "Model": [],
        "AUC-ROC": [],
        "Balanced Accuracy": [],
        "Average Precision (AUPRC)": []
    }
    
    gender_results = {
        "Model": [],
        "Gender": [],
        "Accuracy": [],
        "F1": []
    }
    
    print(f"\n{'='*70}")
    print("RUNNING LOGO VALIDATION WITH GRID SEARCH")
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
            
            print(f"  Best parameters: {best_params}")
            
            # Overall performance evaluation
            auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, 
                                        scoring='roc_auc') * 100
            balanced_acc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf,
                                                 scoring='balanced_accuracy') * 100
            auprc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf,
                                          scoring='average_precision') * 100
            
            results["Model"].append(model_name)
            results["AUC-ROC"].append(f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
            results["Balanced Accuracy"].append(f"{np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
            results["Average Precision (AUPRC)"].append(f"{np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
            
            print(f"  AUPRC: {np.mean(auprc_scores):.2f} ± {np.std(auprc_scores):.2f}")
            print(f"  AUC-ROC: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
            print(f"  Balanced Acc: {np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
            
            # Save interim results
            pd.DataFrame(results).to_csv(INTERIM_RESULTS_PATH, index=False)
            
            # Gender-stratified evaluation
            print(f"  Evaluating gender-stratified performance...")
            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=groups)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                gender_test = gender_labels[test_idx]
                
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                
                for gender in np.unique(gender_test):
                    idx = np.where(gender_test == gender)[0]
                    if len(idx) == 0:
                        continue
                    
                    acc = accuracy_score(y_test[idx], y_pred[idx])
                    f1 = f1_score(y_test[idx], y_pred[idx], average='binary', zero_division=0)
                    
                    gender_results["Model"].append(model_name)
                    gender_results["Gender"].append(gender)
                    gender_results["Accuracy"].append(acc)
                    gender_results["F1"].append(f1)
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
    
    return pd.DataFrame(results), pd.DataFrame(gender_results)

#%%
def analyze_gender_performance(gender_df):
    """
    Analyze and display gender-stratified performance.
    
    Parameters:
    -----------
    gender_df : pd.DataFrame
        Gender-stratified performance metrics
    """
    print(f"\n{'='*70}")
    print("GENDER-STRATIFIED PERFORMANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    for model_name in gender_df['Model'].unique():
        model_data = gender_df[gender_df['Model'] == model_name]
        print(f"{model_name}:")
        
        gender_summary = model_data.groupby('Gender').agg({
            'Accuracy': ['mean', 'std', 'count'],
            'F1': ['mean', 'std']
        }).round(3)
        
        print(gender_summary)
        print()

def save_final_results(results_df, gender_df):
    """
    Save final results to CSV files.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Overall model performance
    gender_df : pd.DataFrame
        Gender-stratified performance
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_path = os.path.join(OUTPUT_DIR, FINAL_RESULTS_PATH)
    gender_path = os.path.join(OUTPUT_DIR, 'gender_stratified_performance.csv')
    
    results_df.to_csv(results_path, index=False)
    gender_df.to_csv(gender_path, index=False)
    
    print(f"\nFinal results saved to:")
    print(f"  {results_path}")
    print(f"  {gender_path}")

#%%
if __name__ == "__main__":
    # Run LOGO validation
    results_df, gender_df = run_logo_validation_with_demographics()
    
    print(f"\n{'='*70}")
    print("OVERALL LOGO VALIDATION RESULTS")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    
    # Analyze gender-stratified performance
    analyze_gender_performance(gender_df)
    
    # Save results
    save_final_results(results_df, gender_df)
    
    print(f"\n{'='*70}")
    print("LOGO VALIDATION COMPLETE")
    print(f"{'='*70}")
    print("\nNote: High variance in LOGO results is expected due to small")
    print("sample sizes per participant. Consider using stratified group")
    print("k-fold for more stable performance estimates.")