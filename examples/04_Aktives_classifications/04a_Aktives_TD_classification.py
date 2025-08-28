#%% imports for ML evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('../../data/Aktives/extracted_features/ppg_features_combined.csv')

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['participant'].nunique()}")
print(f"Label distribution:\n{data['label'].value_counts()}")
print(f"Samples per participant:\n{data['participant'].value_counts().describe()}")

#%% Feature selection
feats = ['HR_mean','HR_std','SDNN','SDSD','meanNN','medianNN','RMSSD','meanSD','sqi'] # no pnn50/20 used because of data length, optionally include sqi as additional feature
label = 'label'

# One hot encode the cohort column to use as contextual data for the model (optional)
cohort_dummies = pd.get_dummies(data['cohort'], prefix='cohort', drop_first=False)

# Combine original features with cohort dummies
X = pd.concat([data[feats], cohort_dummies], axis=1).values
y = data[label].values
groups = data['participant'].values

print(f"Total features: {X.shape[1]} ({len(feats)} HRV + {len(cohort_dummies.columns)} cohort)")

#%% Normalize features first
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#%% Simple sample-level 70/30 split (like the paper)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nSample-Level 70/30 Split:")
print(f"Train samples: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")
print(f"Train label distribution: {np.bincount(y_train)} (ratio: {np.sum(y_train)/len(y_train):.3f})")
print(f"Test label distribution: {np.bincount(y_test)} (ratio: {np.sum(y_test)/len(y_test):.3f})")

#%% Parameter grids
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

#%% Define models (same as your LOGO code)
models = {
    "Random Forest": RandomForestClassifier(random_state=0, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

#%% Initialize results storage (with standard deviations)
results = {
    "Model": [],
    "AUC": [],
    "AUC_SD": [],
    "Accuracy": [],
    "Accuracy_SD": [],
    "F1 Score": [],
    "F1_SD": [],
    "Best Parameters": []
}

#%% Model evaluation loop (structured like your LOGO code)
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    try:
        # Get best parameters using inner CV
        print("Finding best parameters with inner cross-validation...")
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grids[model_name], 
            cv=inner_cv, 
            scoring='roc_auc', 
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Calculate cross-validation scores for standard deviations
        print("Calculating cross-validation scores for standard deviations...")
        cv_scores = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        
        # Get CV scores for each metric
        from sklearn.metrics import make_scorer
        auc_scores = cross_val_score(best_model, X_train, y_train, cv=cv_scores, scoring='roc_auc')
        acc_scores = cross_val_score(best_model, X_train, y_train, cv=cv_scores, scoring='accuracy')
        f1_scores = cross_val_score(best_model, X_train, y_train, cv=cv_scores, scoring='f1_weighted')
        
        # Calculate means and standard deviations
        auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
        acc_mean, acc_std = np.mean(acc_scores), np.std(acc_scores)
        f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
        
        # Also get test set performance for comparison
        print("Making predictions on test set...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store TEST SET results (for paper comparison)
        results["Model"].append(model_name)
        results["AUC"].append(f"{test_auc*100:.2f}")
        results["Accuracy"].append(f"{test_acc*100:.2f}")  
        results["F1 Score"].append(f"{test_f1*100:.2f}")
        results["AUC_SD"].append(f"{auc_std*100:.3f}")  # Keep CV SD for reference
        results["Accuracy_SD"].append(f"{acc_std*100:.3f}")
        results["F1_SD"].append(f"{f1_std*100:.3f}")
        results["Best Parameters"].append(str(best_params))
        
        print(f"\n{model_name} Results:")
        print(f"  Cross-Validation: AUC={auc_mean*100:.2f}±{auc_std*100:.2f}, Acc={acc_mean*100:.2f}±{acc_std*100:.2f}, F1={f1_mean*100:.2f}±{f1_std*100:.2f}")
        print(f"  Test Set: AUC={test_auc*100:.2f}, Acc={test_acc*100:.2f}, F1={test_f1*100:.2f}")
        print(f"  → Using Test Set results for paper comparison")
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        results["Model"].append(model_name)
        results["AUC"].append("ERROR")
        results["AUC_SD"].append("ERROR")
        results["Accuracy"].append("ERROR")
        results["Accuracy_SD"].append("ERROR")
        results["F1 Score"].append("ERROR")
        results["F1_SD"].append("ERROR")
        results["Best Parameters"].append("ERROR")

#%% Display final results (same format as LOGO)
print(f"\n{'='*70}")
print("FINAL 70/30 SPLIT RESULTS")
print(f"{'='*70}")

# Convert to DataFrame and display TEST SET results (matching paper format)
results_df = pd.DataFrame(results)

print("TEST SET RESULTS (for direct comparison with paper):")
print(f"{'Model':<20} {'AUC':<8} {'Accuracy':<10} {'F1 Score':<10}")
print("-" * 55)
for _, row in results_df.iterrows():
    if row['AUC'] != 'ERROR':
        print(f"{row['Model']:<20} {row['AUC']:<8} {row['Accuracy']:<10} {row['F1 Score']:<10}")
    else:
        print(f"{row['Model']:<20} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")

#%% Analysis and comparison (like LOGO analysis)
print(f"\n{'='*60}")
print("PERFORMANCE ANALYSIS")
print(f"{'='*60}")

# Find best performing models
valid_results = results_df[results_df['AUC'] != 'ERROR'].copy()
if len(valid_results) > 0:
    # Convert to numeric for analysis
    valid_results['AUC_numeric'] = valid_results['AUC'].astype(float)
    valid_results['Accuracy_numeric'] = valid_results['Accuracy'].astype(float)
    valid_results['F1_numeric'] = valid_results['F1 Score'].astype(float)
    
    # Sort by AUC
    valid_results_sorted = valid_results.sort_values('AUC_numeric', ascending=False)
    
    print("Models ranked by test set AUC performance:")
    for i, (_, row) in enumerate(valid_results_sorted.iterrows()):
        print(f"{i+1}. {row['Model']}: AUC={row['AUC']}, Acc={row['Accuracy']}, F1={row['F1 Score']}")
    
    # Best results
    best_auc_model = valid_results_sorted.iloc[0]
    best_acc_model = valid_results.loc[valid_results['Accuracy_numeric'].idxmax()]
    best_f1_model = valid_results.loc[valid_results['F1_numeric'].idxmax()]
    
    print(f"\nBest Test Set Results:")
    print(f"  Best AUC: {best_auc_model['Model']} ({best_auc_model['AUC']})")
    print(f"  Best Accuracy: {best_acc_model['Model']} ({best_acc_model['Accuracy']})")
    print(f"  Best F1: {best_f1_model['Model']} ({best_f1_model['F1 Score']})")

#%% Save results
results_df.to_csv('aktives_70_30_split_results.csv', index=False)
print(f"\n✓ Results saved to 'aktives_70_30_split_results.csv'")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE!")
print(f"{'='*60}")
print("Results should be much better than the previous participant-level split")
print("You can now use this table for your manuscript comparison")