#%% imports
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, cross_val_score, StratifiedKFold, LeaveOneOut, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb
import shap

#%% Configuration
# Task Configuration
TASK = 'sleep'  # Options: 'stress' or 'sleep'
FEATURE_SET = 'td_best_surveys_sqi'  # Options: 'td_best', 'td_best_sqi', 'td_best_surveys', 'td_best_surveys_sqi', 'all_td'

# File paths
DATA_PATH = '../data/Wellby/Wellby_all_subjects_features.csv'
RESULTS_DIR = '../results/Wellby/TD_classification/'

# Feature Set Definitions
feature_sets = {
    'stress': {
        'td_best': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50'],
        'td_best_sqi': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','SQI'],
        'td_best_surveys': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','PSS','PSQI','EPOCH'],
        'td_best_surveys_sqi': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','PSS','PSQI','EPOCH','SQI'],
        'all_td': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','meanSD','SDSD','PSS','PSQI','EPOCH','SQI']
    },
    'sleep': {
        'td_best': ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50'],
        'td_best_sqi': ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','SQI'],
        'td_best_surveys': ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','PSS','PSQI','EPOCH'],
        'td_best_surveys_sqi': ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','PSS','PSQI','EPOCH','SQI'],
        'all_td': ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','meanSD','SDSD','PSS','PSQI','EPOCH','SQI']
    }
}

# Generate configuration based on inputs
label = f'{TASK}_binary'
feats = feature_sets[TASK][FEATURE_SET]
output_filename = f'Wellby_{TASK}_{FEATURE_SET}_eval.csv'

print(f"Configuration:")
print(f"Task: {TASK}")
print(f"Feature Set: {FEATURE_SET}")
print(f"Features: {feats}")
print(f"Label: {label}")
print(f"Output file: {output_filename}")
print("-" * 50)

#%% imports
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import os

# Load data
data = pd.read_csv(DATA_PATH)

# Separate features and labels
X = data[feats].values
y = data[label].values
groups = data['Participant'].values

# Normalize the selected features
sc = StandardScaler()
X = sc.fit_transform(X)

# group k-fold 
kf = StratifiedGroupKFold(n_splits=3)

#%% Define parameter grid for models
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
        'solver': ['lsqr'],  # Fixed to avoid shrinkage incompatibility
        'shrinkage': [None, 'auto']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['linear','rbf']
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
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

#%% Perform Grid Search
# Initialize results storage
results = {
    "Model": [],
    "Average Precision (AUPOC)": [],
    "AUC": [],
    "Balanced Accuracy": [],
    "Best Parameters": []
}

# Perform grid search for each model and evaluate
for model_name, model in models.items():
    print(f"Training {model_name}...")
    try:
        # Perform Grid Search
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grids[model_name], 
            cv=kf, 
            scoring='average_precision', 
            n_jobs=2
        )
        grid_search.fit(X, y, groups=groups)
        
        # Retrieve best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Evaluate the best model across metrics (fixed cross_val_score calls)
        auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='roc_auc')*100
        balanced_acc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='balanced_accuracy')*100
        aupoc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='average_precision')*100

        # Append evaluation results
        results["Model"].append(model_name)
        results["Average Precision (AUPOC)"].append(f"{np.mean(aupoc_scores):.2f} ± {np.std(aupoc_scores):.2f}")
        results["AUC"].append(f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
        results["Balanced Accuracy"].append(f"{np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
        results["Best Parameters"].append(best_params)

    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Convert final results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

# Save the results to a CSV file
# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save with dynamic filename
output_path = os.path.join(RESULTS_DIR, output_filename)
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
# %%
