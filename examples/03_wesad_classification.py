# LOGO cross validation of the WESAD feature set adapted from Heo et al. 2021
# applied to the CNN and TD combined feature set

# + Code for Table 5 in paper
import pandas as pd
import numpy as np
import csv
from scipy import stats  # ADDED for confidence intervals

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn import metrics  
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix  

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','subject','label']

WINDOW_SIZE = '120'

subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]


# +
from collections import Counter

def calculate_confidence_intervals(scores, confidence_level=0.95):
    """
    Calculate confidence intervals for cross-validation results
    
    Parameters:
    scores: array-like, performance scores from cross-validation
    confidence_level: float, confidence level (default 0.95 for 95% CI)
    
    Returns:
    dict with mean, std, CI lower bound, CI upper bound
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


def read_csv(path, feats, testset_num):
    print("testset num: ",testset_num)
    df = pd.read_csv(path, index_col = 0)
    
    df = df[feats]

    train_df = df.loc[df['subject'] != testset_num]
    test_df =  df.loc[df['subject'] == testset_num]

    # Debugging train-test split
    print("Train subjects:", train_df['subject'].unique())
    print("Test subject:", test_df['subject'].unique())

    del train_df['subject']
    del test_df['subject']
    del df['subject']

    
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values   
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values    
    
    return df, X_train, y_train, X_test, y_test
# -



# # Machine learning models


def RF_model(X_train, y_train, X_test, y_test):
    
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def AB_model(X_train, y_train, X_test, y_test):
    
    model = AdaBoostClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def KN_model(X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy

def LDA_model(X_train, y_train, X_test, y_test):
    
    model = LinearDiscriminantAnalysis()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    
    return AUC, F1, accuracy

def SVM_model(X_train, y_train, X_test, y_test):
    
    model = svm.SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    
    return AUC, F1, accuracy


def GB_model(X_train, y_train, X_test, y_test):
    
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1= f1_score(y_test, y_pred)
    
    
    return AUC, F1, accuracy

# +

    
path = '../data/WESAD_all_subjects_TD_features.csv'
result_path_all = '../results/WESAD/WESAD_TD_features_eval.csv'
result_path_ci = '../results/WESAD/WESAD_TD_features_eval_with_ci.csv'

RF_AUC, RF_F1, RF_ACC = [], [], []
AB_AUC, AB_F1, AB_ACC = [], [], []
KN_AUC, KN_F1, KN_ACC = [], [], []
LDA_AUC, LDA_F1, LDA_ACC = [], [], []
SVM_AUC, SVM_F1, SVM_ACC = [], [], []
GB_AUC, GB_F1, GB_ACC = [], [], []

for sub in subjects:

    df, X_train, y_train, X_test, y_test = read_csv(path, feats, sub)
    df.fillna(0)
    # Normalization
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    auc_rf, f1_rf, acc_rf = RF_model(X_train, y_train, X_test, y_test)
    auc_ab, f1_ab, acc_ab = AB_model(X_train, y_train, X_test, y_test)
    auc_kn, f1_kn, acc_kn = KN_model(X_train, y_train, X_test, y_test)
    auc_lda, f1_lda, acc_lda = LDA_model(X_train, y_train, X_test, y_test)
    auc_svm, f1_svm, acc_svm = SVM_model(X_train, y_train, X_test, y_test)
    auc_gb, f1_gb, acc_gb = GB_model(X_train, y_train, X_test, y_test)

    RF_AUC.append(auc_rf*100)
    RF_F1.append(f1_rf*100)
    RF_ACC.append(acc_rf*100)
    AB_AUC.append(auc_ab*100)
    AB_F1.append(f1_ab*100)
    AB_ACC.append(acc_ab*100)
    KN_AUC.append(auc_kn*100)
    KN_F1.append(f1_kn*100)
    KN_ACC.append(acc_kn*100)
    LDA_AUC.append(auc_lda*100)
    LDA_F1.append(f1_lda*100)
    LDA_ACC.append(acc_lda*100)
    SVM_AUC.append(auc_svm*100)
    SVM_F1.append(f1_svm*100)
    SVM_ACC.append(acc_svm*100)
    GB_AUC.append(auc_gb*100)
    GB_F1.append(f1_gb*100)
    GB_ACC.append(acc_gb*100)

with open(result_path_all, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['subject','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17','total'])
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

with open(result_path_ci, 'w', newline='') as file:
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


print("\n" + "="*80)
print("RESULTS FOR YOUR PAPER (Table V)")
print("="*80)

print("\nAUC-ROC Results:")
print("-" * 40)
for alg, scores in [('RF', RF_AUC), ('AB', AB_AUC), ('kNN', KN_AUC), ('LDA', LDA_AUC), ('SVM', SVM_AUC), ('GB', GB_AUC)]:
    ci_results = calculate_confidence_intervals(scores)
    print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} (95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

print("\nF1 Score Results:")
print("-" * 40)
for alg, scores in [('RF', RF_F1), ('AB', AB_F1), ('kNN', KN_F1), ('LDA', LDA_F1), ('SVM', SVM_F1), ('GB', GB_F1)]:
    ci_results = calculate_confidence_intervals(scores)
    print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} (95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

print("\nAccuracy Results:")
print("-" * 40)
for alg, scores in [('RF', RF_ACC), ('AB', AB_ACC), ('kNN', KN_ACC), ('LDA', LDA_ACC), ('SVM', SVM_ACC), ('GB', GB_ACC)]:
    ci_results = calculate_confidence_intervals(scores)
    print(f"{alg}: {ci_results['mean']:.2f} ± {ci_results['std']:.2f} (95% CI: {ci_results['ci_lower']:.2f}, {ci_results['ci_upper']:.2f})")

print("\nDONE: Results saved to both original and CI files")




