# LOGO cross validation with CNN + TD combined feature set
# Adapted from original WESAD evaluation to work with combined features

import pandas as pd
import numpy as np
import csv
from scipy import stats

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn import metrics  
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix  

# Configuration
COMBINED_DATA_PATH = '../data/all_subjects_WESAD_hybrid_features.csv'
RESULTS_PATH_ALL = '../results/WESAD/WESAD_hybrid_features_eval.csv'
RESULTS_PATH_CI = '../results/WESAD/WESAD_hybrid_features_eval_with_ci.csv'

# Subject list (same as before)
subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

def get_feature_columns(df):
    """
    Automatically determine which columns are features vs. metadata
    """
    # Columns that are NOT features
    non_feature_cols = ['subject_id', 'label']
    
    # Get all feature columns
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Non-feature columns: {non_feature_cols}")
    print(f"First 10 feature columns: {feature_cols[:10]}")
    print(f"Last 10 feature columns: {feature_cols[-10:]}")
    
    return feature_cols

def calculate_confidence_intervals(scores, confidence_level=0.95):
    """
    Calculate confidence intervals for cross-validation results
    """
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # Sample standard deviation
    
    # Calculate standard error
    se = std / np.sqrt(n)
    
    # Calculate t-value for given confidence level
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Calculate confidence interval
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se
    
    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def read_combined_csv(path, testset_num):
    """
    Read combined CSV file and split into train/test based on subject
    """
    print(f"Loading data from: {path}")
    print(f"Test subject: S{testset_num}")
    
    # Load the combined dataset
    df = pd.read_csv(path)
    print(f"Total dataset shape: {df.shape}")
    
    # Get feature columns automatically
    feature_cols = get_feature_columns(df)
    
    # Ensure we have the required columns
    if 'subject_id' not in df.columns:
        raise ValueError("subject_id column not found in dataset")
    if 'label' not in df.columns:
        raise ValueError("label column not found in dataset")
    
    # Split by subject
    train_df = df[df['subject_id'] != testset_num].copy()
    test_df = df[df['subject_id'] == testset_num].copy()
    
    # Debugging train-test split
    print(f"Train subjects: {sorted(train_df['subject_id'].unique())}")
    print(f"Test subject: {sorted(test_df['subject_id'].unique())}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Check if test subject has data
    if len(test_df) == 0:
        print(f"WARNING: No data found for test subject S{testset_num}")
        return None, None, None, None, None, None
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    return df, X_train, y_train, X_test, y_test, feature_cols

# Machine learning models

def RF_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def AB_model(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def KN_model(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def LDA_model(X_train, y_train, X_test, y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def SVM_model(X_train, y_train, X_test, y_test):
    model = svm.SVC(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def GB_model(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

# Main evaluation loop
def run_evaluation():
    print("="*80)
    print("COMBINED CNN + TD FEATURES EVALUATION")
    print("="*80)
    
    # Initialize result lists
    RF_AUC, RF_F1, RF_ACC = [], [], []
    AB_AUC, AB_F1, AB_ACC = [], [], []
    KN_AUC, KN_F1, KN_ACC = [], [], []
    LDA_AUC, LDA_F1, LDA_ACC = [], [], []
    SVM_AUC, SVM_F1, SVM_ACC = [], [], []
    GB_AUC, GB_F1, GB_ACC = [], [], []
    
    valid_subjects = []
    
    for sub in subjects:
        print(f"\n{'='*40}")
        print(f"Processing Subject S{sub}")
        print(f"{'='*40}")
        
        try:
            # Load and split data
            df, X_train, y_train, X_test, y_test, feature_cols = read_combined_csv(
                COMBINED_DATA_PATH, sub
            )
            
            if X_train is None:
                print(f"Skipping subject S{sub} - no data")
                continue
            
            # Handle missing values
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalization
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            # Run all models
            print("Running models...")
            auc_rf, f1_rf, acc_rf = RF_model(X_train, y_train, X_test, y_test)
            auc_ab, f1_ab, acc_ab = AB_model(X_train, y_train, X_test, y_test)
            auc_kn, f1_kn, acc_kn = KN_model(X_train, y_train, X_test, y_test)
            auc_lda, f1_lda, acc_lda = LDA_model(X_train, y_train, X_test, y_test)
            auc_svm, f1_svm, acc_svm = SVM_model(X_train, y_train, X_test, y_test)
            auc_gb, f1_gb, acc_gb = GB_model(X_train, y_train, X_test, y_test)
            
            # Store results (convert to percentages)
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
            
            valid_subjects.append(sub)
            
            print(f"Subject S{sub} results:")
            print(f"  Best AUC: {max(auc_rf, auc_lda, auc_svm, auc_gb)*100:.2f}%")
            print(f"  Features used: {len(feature_cols)}")
            
        except Exception as e:
            print(f"Error processing subject S{sub}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(valid_subjects)} subjects: {valid_subjects}")
    
    # Save detailed results
    print(f"\nSaving detailed results to: {RESULTS_PATH_ALL}")
    with open(RESULTS_PATH_ALL, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Create header with valid subjects
        header = ['subject'] + [f'S{s}' for s in valid_subjects] + ['mean']
        writer.writerow(header)
        
        # Write results
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
    
    # Save results with confidence intervals
    print(f"Saving CI results to: {RESULTS_PATH_CI}")
    with open(RESULTS_PATH_CI, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        writer.writerow(['Algorithm', 'Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper', 'Formatted_Result'])
        
        # All results with confidence intervals
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
            if len(scores) > 0:
                ci_results = calculate_confidence_intervals(scores)
                formatted_result = f"{ci_results['mean']:.2f} ± {ci_results['std']:.2f} (95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})"
                
                writer.writerow([
                    alg, 
                    metric,
                    f"{ci_results['mean']:.4f}",
                    f"{ci_results['std']:.4f}",
                    f"{ci_results['ci_lower']:.4f}",
                    f"{ci_results['ci_upper']:.4f}",
                    formatted_result
                ])
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY - COMBINED CNN + TD FEATURES")
    print("="*80)
    
    if len(RF_AUC) > 0:
        print("\nAUC-ROC Results:")
        print("-" * 50)
        for alg, scores in [('RF', RF_AUC), ('AB', AB_AUC), ('kNN', KN_AUC), 
                          ('LDA', LDA_AUC), ('SVM', SVM_AUC), ('GB', GB_AUC)]:
            if len(scores) > 0:
                ci_results = calculate_confidence_intervals(scores)
                print(f"{alg:4s}: {ci_results['mean']:6.2f} ± {ci_results['std']:5.2f} "
                      f"(95% CI: {ci_results['ci_lower']:6.2f}, {ci_results['ci_upper']:6.2f})")
        
        print("\nF1 Score Results:")
        print("-" * 50)
        for alg, scores in [('RF', RF_F1), ('AB', AB_F1), ('kNN', KN_F1), 
                          ('LDA', LDA_F1), ('SVM', SVM_F1), ('GB', GB_F1)]:
            if len(scores) > 0:
                ci_results = calculate_confidence_intervals(scores)
                print(f"{alg:4s}: {ci_results['mean']:6.2f} ± {ci_results['std']:5.2f} "
                      f"(95% CI: {ci_results['ci_lower']:6.2f}, {ci_results['ci_upper']:6.2f})")
        
        print("\nAccuracy Results:")
        print("-" * 50)
        for alg, scores in [('RF', RF_ACC), ('AB', AB_ACC), ('kNN', KN_ACC), 
                          ('LDA', LDA_ACC), ('SVM', SVM_ACC), ('GB', GB_ACC)]:
            if len(scores) > 0:
                ci_results = calculate_confidence_intervals(scores)
                print(f"{alg:4s}: {ci_results['mean']:6.2f} ± {ci_results['std']:5.2f} "
                      f"(95% CI: {ci_results['ci_lower']:6.2f}, {ci_results['ci_upper']:6.2f})")
    
    print(f"\nDONE: Results saved to:")
    print(f"  - Detailed: {RESULTS_PATH_ALL}")
    print(f"  - With CI:  {RESULTS_PATH_CI}")

if __name__ == "__main__":
    run_evaluation()