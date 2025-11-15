"""
LOSO Cross-Validation for WESAD Stress Classification
Outlined in Figure 5 and Table V of the paper
10.1109/TAFFC.2025.3628467

Leave-One-Subject-Out (LOSO) cross-validation for stress detection using
time-domain HRV features extracted from PPG signals. Tests six classical ML
algorithms and calculates performance metrics with 95% confidence intervals.

Based on feature set adapted from:
S. Heo, S. Kwon and J. Lee, "Stress Detection With Single PPG Sensor by 
Orchestrating Multiple Denoising and Peak-Detecting Methods," IEEE Access, 
vol. 9, pp. 47777-47785, 2021.

Created: November 2025
Author: Justin Laiti
Claude AI used to generate function docstrings and code structure.
"""
#%% imports
import pandas as pd
import numpy as np
import csv
from scipy import stats
from collections import Counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

#%%
# Configuration
FEATURES = ['HR_mean', 'HR_std', 'meanNN', 'SDNN', 'medianNN', 'meanSD', 
            'SDSD', 'RMSSD', 'pNN20', 'pNN50', 'sqi', 'subject', 'label']
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WINDOW_SIZE = '120'

# TODO: update paths based on local setup
DATA_PATH = '../../data/WESAD_features_merged_bp_time.csv'
RESULTS_PATH = '../../results/WESAD/WESAD_TD_features.csv'
RESULTS_CI_PATH = '../../results/WESAD/WESAD_TD_features_with_ci.csv'

#%%
def calculate_confidence_intervals(scores, confidence_level=0.95):
    """
    Calculate confidence intervals for cross-validation results.
    
    Uses t-distribution to compute confidence intervals appropriate for
    small sample sizes (n=15 subjects in LOSO validation).
    
    Parameters:
    -----------
    scores : array-like
        Performance scores from cross-validation (one per test subject)
    confidence_level : float, default=0.95
        Confidence level for interval calculation
        
    Returns:
    --------
    dict
        Dictionary containing:
        - mean: Mean performance across subjects
        - std: Standard deviation
        - se: Standard error
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
    """
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    
    se = std / np.sqrt(n)
    
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se
    
    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def read_csv(path, feats, testset_num):
    """
    Load and split data for LOSO cross-validation.
    
    Splits data such that one subject is held out for testing while
    all other subjects are used for training.
    
    Parameters:
    -----------
    path : str
        Path to CSV file containing features and labels
    feats : list
        List of feature column names to include
    testset_num : int
        Subject ID to hold out for testing
        
    Returns:
    --------
    tuple
        - df: Full dataframe
        - X_train: Training features
        - y_train: Training labels
        - X_test: Test features
        - y_test: Test labels
    """
    print(f"Test subject: S{testset_num}")
    df = pd.read_csv(path, index_col=0)
    df = df[feats]

    train_df = df.loc[df['subject'] != testset_num]
    test_df = df.loc[df['subject'] == testset_num]

    print(f"  Training on subjects: {sorted(train_df['subject'].unique())}")
    print(f"  Testing on subject: {sorted(test_df['subject'].unique())}")

    X_train = train_df.drop(['subject', 'label'], axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop(['subject', 'label'], axis=1).values
    y_test = test_df['label'].values
    
    return df, X_train, y_train, X_test, y_test

#%%
def RF_model(X_train, y_train, X_test, y_test):
    """Random Forest classifier."""
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

def AB_model(X_train, y_train, X_test, y_test):
    """AdaBoost classifier."""
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

def KN_model(X_train, y_train, X_test, y_test):
    """K-Nearest Neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

def LDA_model(X_train, y_train, X_test, y_test):
    """Linear Discriminant Analysis classifier."""
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

def SVM_model(X_train, y_train, X_test, y_test):
    """Support Vector Machine classifier."""
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

def GB_model(X_train, y_train, X_test, y_test):
    """Gradient Boosting classifier."""
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return (roc_auc_score(y_test, y_pred), 
            f1_score(y_test, y_pred), 
            accuracy_score(y_test, y_pred))

#%%
def run_loso_validation():
    """
    Run LOSO cross-validation on all subjects.
    
    For each subject:
    1. Hold out subject as test set
    2. Train on all other subjects
    3. Standardize features using training set statistics
    4. Evaluate six ML algorithms
    5. Record AUC-ROC, F1-score, and accuracy
    
    Saves two CSV files:
    - Full results with per-subject scores
    - Summary with means, std, and 95% confidence intervals
    """
    
    RF_AUC, RF_F1, RF_ACC = [], [], []
    AB_AUC, AB_F1, AB_ACC = [], [], []
    KN_AUC, KN_F1, KN_ACC = [], [], []
    LDA_AUC, LDA_F1, LDA_ACC = [], [], []
    SVM_AUC, SVM_F1, SVM_ACC = [], [], []
    GB_AUC, GB_F1, GB_ACC = [], [], []

    print("\n" + "="*80)
    print("LOSO CROSS-VALIDATION ON WESAD DATASET")
    print("="*80 + "\n")

    for sub in SUBJECTS:
        df, X_train, y_train, X_test, y_test = read_csv(DATA_PATH, FEATURES, sub)
        df.fillna(0, inplace=True)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        auc_rf, f1_rf, acc_rf = RF_model(X_train, y_train, X_test, y_test)
        auc_ab, f1_ab, acc_ab = AB_model(X_train, y_train, X_test, y_test)
        auc_kn, f1_kn, acc_kn = KN_model(X_train, y_train, X_test, y_test)
        auc_lda, f1_lda, acc_lda = LDA_model(X_train, y_train, X_test, y_test)
        auc_svm, f1_svm, acc_svm = SVM_model(X_train, y_train, X_test, y_test)
        auc_gb, f1_gb, acc_gb = GB_model(X_train, y_train, X_test, y_test)

        RF_AUC.append(auc_rf * 100)
        RF_F1.append(f1_rf * 100)
        RF_ACC.append(acc_rf * 100)
        AB_AUC.append(auc_ab * 100)
        AB_F1.append(f1_ab * 100)
        AB_ACC.append(acc_ab * 100)
        KN_AUC.append(auc_kn * 100)
        KN_F1.append(f1_kn * 100)
        KN_ACC.append(acc_kn * 100)
        LDA_AUC.append(auc_lda * 100)
        LDA_F1.append(f1_lda * 100)
        LDA_ACC.append(acc_lda * 100)
        SVM_AUC.append(auc_svm * 100)
        SVM_F1.append(f1_svm * 100)
        SVM_ACC.append(acc_svm * 100)
        GB_AUC.append(auc_gb * 100)
        GB_F1.append(f1_gb * 100)
        GB_ACC.append(acc_gb * 100)
        
        print()

    with open(RESULTS_PATH, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['subject', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 
                        'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'total'])
        writer.writerow(['RF_AUC'] + RF_AUC + [np.mean(RF_AUC)])
        writer.writerow(['AB_AUC'] + AB_AUC + [np.mean(AB_AUC)])
        writer.writerow(['KN_AUC'] + KN_AUC + [np.mean(KN_AUC)])
        writer.writerow(['LDA_AUC'] + LDA_AUC + [np.mean(LDA_AUC)])
        writer.writerow(['SVM_AUC'] + SVM_AUC + [np.mean(SVM_AUC)])
        writer.writerow(['GB_AUC'] + GB_AUC + [np.mean(GB_AUC)])
        writer.writerow(['RF_F1'] + RF_F1 + [np.mean(RF_F1)])
        writer.writerow(['AB_F1'] + AB_F1 + [np.mean(AB_F1)])
        writer.writerow(['KN_F1'] + KN_F1 + [np.mean(KN_F1)])
        writer.writerow(['LDA_F1'] + LDA_F1 + [np.mean(LDA_F1)])
        writer.writerow(['SVM_F1'] + SVM_F1 + [np.mean(SVM_F1)])
        writer.writerow(['GB_F1'] + GB_F1 + [np.mean(GB_F1)])
        writer.writerow(['RF_ACC'] + RF_ACC + [np.mean(RF_ACC)])
        writer.writerow(['AB_ACC'] + AB_ACC + [np.mean(AB_ACC)])
        writer.writerow(['KN_ACC'] + KN_ACC + [np.mean(KN_ACC)])
        writer.writerow(['LDA_ACC'] + LDA_ACC + [np.mean(LDA_ACC)])
        writer.writerow(['SVM_ACC'] + SVM_ACC + [np.mean(SVM_ACC)])
        writer.writerow(['GB_ACC'] + GB_ACC + [np.mean(GB_ACC)])

    with open(RESULTS_CI_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Algorithm', 'Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper', 'Formatted_Result'])
        
        results_data = [
            ('RF', 'AUC-ROC', RF_AUC),
            ('RF', 'F1', RF_F1),
            ('RF', 'Accuracy', RF_ACC),
            ('AB', 'AUC-ROC', AB_AUC),
            ('AB', 'F1', AB_F1),
            ('AB', 'Accuracy', AB_ACC),
            ('kNN', 'AUC-ROC', KN_AUC),
            ('kNN', 'F1', KN_F1),
            ('kNN', 'Accuracy', KN_ACC),
            ('LDA', 'AUC-ROC', LDA_AUC),
            ('LDA', 'F1', LDA_F1),
            ('LDA', 'Accuracy', LDA_ACC),
            ('SVM', 'AUC-ROC', SVM_AUC),
            ('SVM', 'F1', SVM_F1),
            ('SVM', 'Accuracy', SVM_ACC),
            ('GB', 'AUC-ROC', GB_AUC),
            ('GB', 'F1', GB_F1),
            ('GB', 'Accuracy', GB_ACC)
        ]
        
        for alg, metric, scores in results_data:
            ci_results = calculate_confidence_intervals(scores)
            formatted_result = (f"{ci_results['mean']:.2f} ± {ci_results['std']:.2f} "
                              f"(95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")
            
            writer.writerow([
                alg, 
                metric,
                f"{ci_results['mean']:.4f}",
                f"{ci_results['std']:.4f}",
                f"{ci_results['ci_lower']:.4f}",
                f"{ci_results['ci_upper']:.4f}",
                formatted_result
            ])

    print("\n" + "="*80)
    print("RESULTS FOR TABLE V IN PAPER")
    print("="*80)

    print("\nAUC-ROC Results:")
    print("-" * 40)
    for alg, scores in [('RF', RF_AUC), ('AB', AB_AUC), ('kNN', KN_AUC), 
                        ('LDA', LDA_AUC), ('SVM', SVM_AUC), ('GB', GB_AUC)]:
        ci_results = calculate_confidence_intervals(scores)
        print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} "
              f"(95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

    print("\nF1 Score Results:")
    print("-" * 40)
    for alg, scores in [('RF', RF_F1), ('AB', AB_F1), ('kNN', KN_F1), 
                        ('LDA', LDA_F1), ('SVM', SVM_F1), ('GB', GB_F1)]:
        ci_results = calculate_confidence_intervals(scores)
        print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} "
              f"(95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

    print("\nAccuracy Results:")
    print("-" * 40)
    for alg, scores in [('RF', RF_ACC), ('AB', AB_ACC), ('kNN', KN_ACC), 
                        ('LDA', LDA_ACC), ('SVM', SVM_ACC), ('GB', GB_ACC)]:
        ci_results = calculate_confidence_intervals(scores)
        print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} "
              f"(95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

    print(f"\nResults saved to:")
    print(f"  {RESULTS_PATH}")
    print(f"  {RESULTS_CI_PATH}")

if __name__ == "__main__":
    run_loso_validation()
# %%
