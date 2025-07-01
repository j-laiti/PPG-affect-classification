
#%% imports
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../data/Aktives/extracted_features/ppg_features_combined.csv')
# data = data[~data['participant'].isin(['C3', 'C5', 'C12', 'C8', 'C9', 'C2', 'C16'])]

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['participant'].nunique()}")
print(f"Label distribution:\n{data['label'].value_counts()}")
print(f"Samples per participant:\n{data['participant'].value_counts().describe()}")

#%% all td features
# feats = ['HR_mean','HR_std','meanNN','medianNN','RMSSD','pNN20','pNN50','meanSD','SDSD']
feats =['HR_mean','HR_std','meanNN','medianNN','RMSSD','meanSD','sqi'] # more reliable HRV features for 30 second windows
label = 'label'

#one hot encode the cohort column
cohort_dummies = pd.get_dummies(data['cohort'], prefix='cohort', drop_first=False)

# Combine original features with cohort dummies
X = pd.concat([data[feats], cohort_dummies], axis=1).values
y = data[label].values
groups = data['participant'].values

# Check for participants with only one class (problematic for some metrics)
participant_class_counts = data.groupby('participant')['label'].nunique()
single_class_participants = participant_class_counts[participant_class_counts == 1].index.tolist()
if len(single_class_participants) > 0:
    print(f"\nWarning: {len(single_class_participants)} participants have only one class:")
    for p in single_class_participants[:5]:  # Show first 5
        participant_labels = data[data['participant'] == p]['label'].values
        print(f"  {p}: {participant_labels[0]} only ({len(participant_labels)} samples)")
    if len(single_class_participants) > 5:
        print(f"  ... and {len(single_class_participants) - 5} more")

#%% Normalize the selected features
sc = StandardScaler()
X = sc.fit_transform(X)

# LOGO cross-validation
logo = LeaveOneGroupOut()
print(f"\nLOGO will create {logo.get_n_splits(X, y, groups)} folds (one per participant)")

# Simplified parameter grids for faster LOGO evaluation
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
        'degree': [2, 3, 4],  # for poly kernel
        'coef0': [0.0, 0.1, 0.5, 1.0]  # for poly/sigmoid kernels
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

# define models
models = {
    "Random Forest": RandomForestClassifier(random_state=0, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

# Initialize results storage
results = {
    "Model": [],
    "AUC": [],
    "Accuracy": [],
    "F1 Score": [],
    "Best Parameters": []
}

# Initialize detailed results for analysis
detailed_results = []

# Perform LOGO cross-validation for each model
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} with LOGO...")
    print(f"{'='*50}")
    
    try:
        # Storage for this model's LOGO results
        fold_auc_scores = []
        fold_acc_scores = []
        fold_f1_scores = []
        fold_details = []
        
        # Get best parameters using inner CV (stratified k-fold on full dataset)
        print("Finding best parameters with inner cross-validation...")
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grids[model_name], 
            cv=inner_cv, 
            scoring='roc_auc', 
            n_jobs=1  # Reduced for stability with LOGO
        )
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {best_params}")
        
        # Now perform LOGO with the best model
        print("Performing LOGO cross-validation...")
        fold_count = 0
        
        for train_idx, test_idx in logo.split(X, y, groups):
            fold_count += 1
            
            # Get train and test data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Get the participant being left out
            test_participant = groups[test_idx][0]  # All test indices should have same participant
            
            # Skip if test set has only one class (can't compute some metrics)
            if len(np.unique(y_test)) == 1:
                print(f"  Fold {fold_count} (participant {test_participant}): Skipped - only one class in test set")
                continue
            
            try:
                # Train model
                best_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Store fold results
                fold_auc_scores.append(auc)
                fold_acc_scores.append(acc)
                fold_f1_scores.append(f1)
                
                # Store detailed results
                fold_details.append({
                    'model': model_name,
                    'fold': fold_count,
                    'test_participant': test_participant,
                    'n_test_samples': len(y_test),
                    'test_class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
                    'n_train_samples': len(y_train),
                    'auc': auc,
                    'accuracy': acc,
                    'f1_score': f1
                })
                
                print(f"  Fold {fold_count} (participant {test_participant}): "
                      f"AUC={auc:.3f}, Accuracy={acc:.3f}, F1={f1:.3f}")

            except Exception as fold_error:
                print(f"  Fold {fold_count} (participant {test_participant}): Error - {fold_error}")
                continue
        
        # Calculate summary statistics
        if len(fold_auc_scores) > 0:
            mean_auc = np.mean(fold_auc_scores)
            std_auc = np.std(fold_auc_scores)
            mean_acc = np.mean(fold_acc_scores)
            std_acc = np.std(fold_acc_scores)
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)
            
            # Append summary results
            results["Model"].append(model_name)
            results["AUC"].append(f"{mean_auc:.3f} ± {std_auc:.3f}")
            results["Accuracy"].append(f"{mean_acc:.3f} ± {std_acc:.3f}")
            results["F1 Score"].append(f"{mean_f1:.3f} ± {std_f1:.3f}")
            results["Best Parameters"].append(str(best_params))
            
            # Add detailed results
            detailed_results.extend(fold_details)
            
            print(f"\n{model_name} Summary:")
            print(f"  Valid folds: {len(fold_auc_scores)}/{logo.get_n_splits(X, y, groups)}")
            print(f"  AUC: {mean_auc:.3f} ± {std_auc:.3f}")
            print(f"  ACC: {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"  F1 Score: {mean_f1:.3f} ± {std_f1:.3f}")
            
        else:
            print(f"\n{model_name}: No valid folds completed!")
            results["Model"].append(model_name)
            results["AUC"].append("N/A")
            results["Accuracy"].append("N/A")
            results["F1 Score"].append("N/A")
            results["Best Parameters"].append(str(best_params))
        
        # Save interim results
        pd.DataFrame(results).to_csv('logo_model_evaluations_summary.csv', index=False)
        if detailed_results:
            pd.DataFrame(detailed_results).to_csv('logo_model_evaluations_detailed.csv', index=False)

    except Exception as e:
        print(f"Error with {model_name}: {e}")
        results["Model"].append(model_name)
        results["AUC"].append("ERROR")
        results["Accuracy"].append("ERROR")
        results["F1 Score"].append("ERROR")
        results["Best Parameters"].append("ERROR")

print(f"\n{'='*70}")
print("FINAL LOGO CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

# Convert final results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

#%% Save final results
results_df.to_csv('final_logo_model_evaluations.csv', index=False)
if detailed_results:
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('final_logo_detailed_results.csv', index=False)
    
    print(f"\nDetailed results saved with {len(detailed_results)} fold evaluations")
    print("Files saved:")
    print("  - final_logo_model_evaluations.csv (summary)")
    print("  - final_logo_detailed_results.csv (per-fold details)")

# Additional analysis
if detailed_results:
    detailed_df = pd.DataFrame(detailed_results)
    
    print(f"\n{'='*50}")
    print("ADDITIONAL ANALYSIS")
    print(f"{'='*50}")
    
    # Best performing model
    model_performance = detailed_df.groupby('model').agg({
        'auc': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'auprc': ['mean', 'std']
    }).round(3)
    
    print("\nModel performance summary:")
    print(model_performance)
    
    # Participants that were hardest to predict
    participant_performance = detailed_df.groupby('test_participant').agg({
        'auc': 'mean',
        'balanced_accuracy': 'mean',
        'auprc': 'mean'
    }).round(3)
    
    worst_participants = participant_performance.sort_values('auc').head(5)
    best_participants = participant_performance.sort_values('auc', ascending=False).head(5)
    
    print(f"\nWorst predicted participants (lowest average AUC):")
    print(worst_participants)
    print(f"\nBest predicted participants (highest average AUC):")
    print(best_participants)

print(f"\n{'='*50}")
print("LOGO EVALUATION COMPLETE!")
print(f"{'='*50}")


#%%

#%% Debug script for AUC = 0.0 issues
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../data/Aktives/extracted_features/ppg_features_combined.csv')

print("=== DEBUGGING AUC = 0.0 ISSUES ===\n")

# data_clean = data[~data['participant'].isin(['C3', 'C5', 'C12', 'C8', 'C9', 'C2', 'C16'])]

#%% 1. Check label encoding
print("1. LABEL ENCODING CHECK:")
print(f"Unique labels: {data['label'].unique()}")
print(f"Label value counts:\n{data['label'].value_counts()}")
print(f"Label data type: {data['label'].dtype}")

# Check if labels are strings vs numeric
if data['label'].dtype == 'object':
    print("⚠ WARNING: Labels are strings - this might cause issues!")
    print("Converting to numeric...")
    # Convert string labels to binary
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['label_encoded'] = le.fit_transform(data['label'])
    print(f"Original -> Encoded mapping:")
    for orig, enc in zip(le.classes_, le.transform(le.classes_)):
        print(f"  '{orig}' -> {enc}")
    label_col = 'label_encoded'
else:
    label_col = 'label'

#%% 2. Check class distribution per participant
print(f"\n2. PARTICIPANT CLASS DISTRIBUTION:")
participant_analysis = data.groupby('participant')[label_col].agg(['count', 'nunique', 'mean']).round(3)
participant_analysis.columns = ['total_samples', 'n_classes', 'positive_ratio']

print("Participants with potential issues:")
problem_participants = []

for participant in data['participant'].unique():
    p_data = data[data['participant'] == participant]
    class_counts = p_data[label_col].value_counts()
    n_classes = len(class_counts)
    
    if n_classes == 1:
        print(f"  {participant}: Only 1 class - {dict(class_counts)}")
        problem_participants.append(participant)
    elif class_counts.min() == 1:
        print(f"  {participant}: Very imbalanced - {dict(class_counts)}")
        problem_participants.append(participant)

print(f"\nTotal problem participants: {len(problem_participants)}")

#%% 3. Check for data issues
print(f"\n3. DATA QUALITY CHECKS:")
feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','meanSD','SDSD']

# Check for missing values
missing_data = data[feats].isnull().sum()
if missing_data.sum() > 0:
    print("⚠ Missing values found:")
    print(missing_data[missing_data > 0])
else:
    print("✓ No missing values in features")

# Check for infinite values
inf_data = np.isinf(data[feats]).sum()
if inf_data.sum() > 0:
    print("⚠ Infinite values found:")
    print(inf_data[inf_data > 0])
else:
    print("✓ No infinite values in features")

# Check feature statistics
print(f"\nFeature statistics:")
print(data[feats].describe())

#%% 4. Test a single problematic fold
print(f"\n4. DETAILED SINGLE FOLD ANALYSIS:")

X = data[feats].values
y = data[label_col].values
groups = data['participant'].values

# Normalize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Set up LOGO
logo = LeaveOneGroupOut()

# Find a fold that might be problematic
model = RandomForestClassifier(random_state=0, class_weight='balanced', n_estimators=50)

print("Analyzing first few folds in detail:")
fold_count = 0
for train_idx, test_idx in logo.split(X, y, groups):
    fold_count += 1
    if fold_count > 3:  # Only check first 3 folds
        break
        
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_participant = groups[test_idx][0]
    
    print(f"\n--- Fold {fold_count} (Participant: {test_participant}) ---")
    print(f"Training set: {len(X_train)} samples")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test set: {len(X_test)} samples")
    print(f"  Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Check if test set has both classes
    if len(np.unique(y_test)) == 1:
        print("  ⚠ Test set has only one class - will skip")
        continue
        
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print(f"  Predictions: {y_pred}")
        print(f"  True labels: {y_test}")
        print(f"  Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"  Prediction probabilities:\n{y_pred_proba}")
        
        # Check if we have probabilities for both classes
        if y_pred_proba.shape[1] < 2:
            print("  ❌ ERROR: Model only predicting one class!")
            continue
            
        # Calculate metrics with detailed output
        try:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"  Results:")
            print(f"    AUC: {auc:.3f}")
            print(f"    Accuracy: {accuracy:.3f}")
            print(f"    F1: {f1:.3f}")
            
            if auc == 0.0:
                print("  ❌ AUC = 0.0 detected!")
                print("  Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                print(f"    {cm}")
                
                # Check if model is predicting opposite of truth
                print("  Checking if predictions are inverted...")
                inverted_auc = roc_auc_score(y_test, 1 - y_pred_proba[:, 1])
                print(f"    Inverted AUC (1 - proba): {inverted_auc:.3f}")
                
        except Exception as metric_error:
            print(f"  ❌ Error calculating metrics: {metric_error}")
            
    except Exception as fold_error:
        print(f"  ❌ Error in fold: {fold_error}")

#%% 5. Check label consistency
print(f"\n5. LABEL CONSISTENCY CHECK:")

# Check if there are any systematic issues
print("Checking for label encoding issues...")

# Original labels
orig_labels = data['label'].unique()
print(f"Original unique labels: {orig_labels}")

# If we encoded labels, check mapping
if 'label_encoded' in data.columns:
    mapping_check = data[['label', 'label_encoded']].drop_duplicates().sort_values('label_encoded')
    print("Label mapping:")
    print(mapping_check)

#%% 6. Suggested fixes
print(f"\n6. SUGGESTED FIXES:")
print("Based on the analysis above, try these fixes:")
print("1. If labels are strings, ensure proper encoding (0/1 or False/True)")
print("2. Remove participants with only one class entirely")
print("3. Check for class imbalance issues in individual folds")
print("4. Verify that your 'label' column contains the right values")
print("5. Consider using StratifiedGroupKFold instead of LOGO if too many single-class participants")

# Quick fix suggestion
if len(problem_participants) > 0:
    print(f"\nQUICK FIX: Try removing the {len(problem_participants)} problematic participants:")
    print("Add this line before your analysis:")
    print(f"data_clean = data[~data['participant'].isin({problem_participants})]")
# %%
