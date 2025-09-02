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
from scipy import stats
warnings.filterwarnings('ignore')

# Load hybrid features data
data = pd.read_csv('../../data/AKTIVES_hybrid_features/all_subjects_AKTIVES_hybrid_features.csv')

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['participant'].nunique()}")
print(f"Label distribution:\n{data['label'].value_counts()}")
print(f"Samples per participant:\n{data['participant'].value_counts().describe()}")
print(f"Cohort distribution:\n{data['cohort'].value_counts()}")

#%% Feature selection - Combined CNN + TD features
# Identify CNN features
cnn_features = [col for col in data.columns if col.startswith('cnn_feature_')]
print(f"CNN features found: {len(cnn_features)}")

# Identify TD features
metadata_cols = ['participant', 'game', 'cohort', 'label', 'interval_start', 'interval_end', 'pNN50', 'pNN20'] # remove pNN20 and pNN50 because of short window size
td_features = [col for col in data.columns if col not in cnn_features and col not in metadata_cols]

print(f"TD features found: {len(td_features)}")
print(f"TD feature names: {td_features}")

# Combine all features
all_features = cnn_features + td_features
label = 'label'

# One hot encode the cohort column
cohort_dummies = pd.get_dummies(data['cohort'], prefix='cohort', drop_first=False)
print(f"Cohort dummy features: {len(cohort_dummies.columns)} ({list(cohort_dummies.columns)})")

# Combine CNN + TD features with cohort dummies
X = pd.concat([data[all_features], cohort_dummies], axis=1).values
y = data[label].values
groups = data['participant'].values

print(f"Total features: {X.shape[1]} ({len(cnn_features)} CNN + {len(td_features)} TD + {len(cohort_dummies.columns)} cohort)")

#%% Normalize features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)


#%% Simple sample-level 70/30 split (like the paper)

def calculate_confidence_intervals(values, confidence=0.95):
    """Calculate confidence intervals for a list of values"""
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

n_splits = 10
all_results = {}  # Dictionary to store results for each model across splits

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

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=0, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

#%% Model evaluation loop

for model_name in models.keys():
    all_results[model_name] = {
        'auc_scores': [],
        'accuracy_scores': [],
        'f1_scores': []
    }

for split_i in range(n_splits):
    print(f"\n=== Split {split_i+1}/{n_splits} ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42+split_i, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

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
            
            # Also get test set performance for comparison
            print("Making predictions on test set...")
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            test_auc = roc_auc_score(y_test, y_pred_proba)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
        
            # Store results for this split
            all_results[model_name]['auc_scores'].append(test_auc * 100)
            all_results[model_name]['accuracy_scores'].append(test_acc * 100)
            all_results[model_name]['f1_scores'].append(test_f1 * 100)
            
            print(f"  {model_name}: AUC={test_auc*100:.2f}, Acc={test_acc*100:.2f}, F1={test_f1*100:.2f}")
        
        except Exception as e:
            print(f"Error with {model_name}: {e}")

# Calculate confidence intervals across all splits
print(f"\n{'='*70}")
print("FINAL RESULTS ACROSS ALL SPLITS WITH CONFIDENCE INTERVALS")
print(f"{'='*70}")

final_results = []

for model_name in models.keys():
    if len(all_results[model_name]['auc_scores']) > 0:
        # Calculate CIs for each metric
        auc_mean, auc_std, auc_ci_low, auc_ci_high = calculate_confidence_intervals(all_results[model_name]['auc_scores'])
        acc_mean, acc_std, acc_ci_low, acc_ci_high = calculate_confidence_intervals(all_results[model_name]['accuracy_scores'])
        f1_mean, f1_std, f1_ci_low, f1_ci_high = calculate_confidence_intervals(all_results[model_name]['f1_scores'])
        
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
        print(f"  AUC:      {auc_mean:.2f} ± {auc_std:.2f} ({auc_ci_low:.2f}, {auc_ci_high:.2f})")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f} ({acc_ci_low:.2f}, {acc_ci_high:.2f})")
        print(f"  F1 Score: {f1_mean:.2f} ± {f1_std:.2f} ({f1_ci_low:.2f}, {f1_ci_high:.2f})")
        print()

# Save results
results_df = pd.DataFrame(final_results)
results_df.to_csv('aktives_hybrid_multiple_splits_with_ci.csv', index=False)
print(f"Results saved to 'aktives_hybrid_multiple_splits_with_ci.csv'")

# Find best performing model
best_auc_idx = results_df['AUC_Mean'].idxmax()
best_model = results_df.loc[best_auc_idx]
print(f"Best performing model (by AUC): {best_model['Algorithm']}")
print(f"AUC: {best_model['AUC_Formatted']}")

#%% Display final results
print(f"\n{'='*70}")
print("FINAL AKTIVES HYBRID FEATURES RESULTS (CNN + TD)")
print(f"{'='*70}")

# Convert to DataFrame and display test set results
results_df = pd.DataFrame(results)

print(f"{'Model':<20} {'AUC':<8} {'Accuracy':<10} {'F1 Score':<10}")
print("-" * 55)
for _, row in results_df.iterrows():
    if row['AUC'] != 'ERROR':
        print(f"{row['Model']:<20} {row['AUC']:<8} {row['Accuracy']:<10} {row['F1 Score']:<10}")
    else:
        print(f"{row['Model']:<20} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")

#%% Save results
results_df.to_csv('../../results/Aktives/aktives_hybrid_features_results.csv', index=False)
print(f"\n✓ Results saved to 'aktives_hybrid_features_results.csv'")
# %%
