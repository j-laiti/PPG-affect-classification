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

# Load data
data = pd.read_csv('data/features/combined_sim_features_3inc_out.csv')

# Define the features and label
# all features
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','HF',
#          'total_power','SD1','SD2','pA','pQ','shanEn','PSS','PSQI','EPOCH','SQI','SNR', 'SNR_freq']

label = 'stress_binary'

# td features (best 8 for stress)
feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50']
# td features (best 8 for stress) + SQI
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','SQI']
# td features (best 8 for stress) + surveys
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','PSS','PSQI','EPOCH']
# td features (best 8 for stress) + surveys + SQI
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','PSS','PSQI','EPOCH','SQI']
# td features (best 8 for sleep) + SQI
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50']
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','SQI']
# td features (best 8 for sleep) + surveys
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','PSS','PSQI','EPOCH']
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','pNN20','pNN50','PSS','PSQI','EPOCH','SQI']
# all td
# feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','meanSD','SDSD','PSS','PSQI','EPOCH','SQI']

# Add schools
# data = pd.get_dummies(data, columns=['School'])
# feats.extend([col for col in data.columns if col.startswith('School')])

# # Add race
# data = pd.get_dummies(data, columns=['Race/ethnicity'])
# feats.extend([col for col in data.columns if col.startswith('Race/ethnicity')])

# # Add gender
# data = pd.get_dummies(data, columns=['Gender'])
# feats.extend([col for col in data.columns if col.startswith('Gender')])

# Add activity
# activity_types = list(range(8))  # Adjust this range based on your actual activity types
# for activity in activity_types:
#     data[f'Activity_{activity}'] = data['Activity'].apply(lambda x: 1 if activity in eval(x) else 0)
# feats.extend([f'Activity_{activity}' for activity in activity_types])

# Separate features and labels
X = data[feats].values
y = data[label].values
groups = data['Participant'].values

# # Filter participants with more than one row of data
# participant_counts = data['Participant'].value_counts()
# participants_to_keep = participant_counts[participant_counts > 0].index

# # Keep only the rows corresponding to these participants
# data_filtered = data[data['Participant'].isin(participants_to_keep)]

# # Separate features and labels
# X = data_filtered[feats].values
# y = data_filtered[label].values
# groups = data_filtered['Participant'].values


# Normalize the selected features
sc = StandardScaler()
X = sc.fit_transform(X)

# group k-fold 
kf = StratifiedGroupKFold(n_splits=3)

# leave on group out
from sklearn.model_selection import BaseCrossValidator

# This only includes groups with both classes
# class StratifiedLeaveOneGroupOut(BaseCrossValidator):
#     def __init__(self, groups, y):
#         self.groups = groups
#         self.y = y
#         self.valid_splits = 0

#     def split(self, X, y=None, groups=None):
#         unique_groups = np.unique(self.groups)
#         for test_group in unique_groups:
#             train_idx = np.where(self.groups != test_group)[0]
#             test_idx = np.where(self.groups == test_group)[0]
#             if len(np.unique(self.y[train_idx])) > 1 and len(np.unique(self.y[test_idx])) > 1:
#                 self.valid_splits += 1
#                 yield train_idx, test_idx

#     def get_n_splits(self, X=None, y=None, groups=None):
#         # Count valid splits by resetting and iterating over all groups
#         self.valid_splits = 0
#         list(self.split(X, y, groups))
#         print(f"Number of valid LOGO splits: {self.valid_splits}")
#         return self.valid_splits

#  # this includes groups that dont have both classes   
# class RelaxedLeaveOneGroupOut(BaseCrossValidator):
#     def __init__(self, groups):
#         self.groups = groups

#     def split(self, X, y=None, groups=None):
#         unique_groups = np.unique(self.groups)
#         for test_group in unique_groups:
#             train_idx = np.where(self.groups != test_group)[0]
#             test_idx = np.where(self.groups == test_group)[0]
#             yield train_idx, test_idx

#     def get_n_splits(self, X=None, y=None, groups=None):
#         return len(np.unique(self.groups))

# # Instantiate
# kf = RelaxedLeaveOneGroupOut(groups=groups)


# Instantiate the custom cross-validator
# kf = StratifiedLeaveOneGroupOut(groups=groups, y=y)

# # Print the number of valid splits before running Grid Search
# n_splits = kf.get_n_splits(X, y, groups)
# print(f"Total valid LOGO validations that will be performed: {n_splits}")


#%% Define parameter grid for SVM
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
        'solver': ['lsqr', 'eigen'],
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
    },
    "LightGBM": {
        'n_estimators': [20, 50],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [7, 15],
        'min_child_samples': [5, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8],
        'boosting_type': ['gbdt']
    }
    ,
    "XGBoost": {
        'n_estimators': [20, 50],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.5, 0.8],
        'colsample_bytree': [0.5, 0.8],
        'objective': ['binary:logistic']
    }

}


# define models
models = {
    "Random Forest": RandomForestClassifier(random_state=0, class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "K-Neighbors": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": svm.SVC(probability=True, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "LightGBM": lgb.LGBMClassifier(random_state=0, is_unbalance=True),
    "XGBoost": xgb.XGBClassifier(random_state=0)  # Adjust scale_pos_weight dynamically
}


#%% Perform Grid Search
from collections import defaultdict

# Initialize results storage
results = {
    "Model": [],
    "AUC": [],
    # "F1": [],
    # "Accuracy": [],
    "Balanced Accuracy": [],
    "Average Precision (AUPOC)": []
    # "Best Parameters": []
}

gender_results = {
    "Model": [],
    "Gender": [],
    "Accuracy": [],
    "F1": []
}

# Pull gender labels from original dataset
gender_labels = data['Gender'].values


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

        # Add custom threshold tuning (if applicable)
                # Compute SHAP values only for Gradient Boosting (or other specific models)
        # if model_name in ["Gradient Boosting"]:
        #     print(f"Computing SHAP values for {model_name}...")
        #     explainer = shap.Explainer(best_model, X)
        #     shap_values = explainer(X)

        #     # SHAP summary plot
        #     shap.summary_plot(shap_values, X, feature_names=feats)

        #     # Optional: Save SHAP values for future analysis
        #     shap_importance = pd.DataFrame({
        #         "Feature": feats,
        #         "Mean SHAP Value": abs(shap_values.values).mean(axis=0)
        #     }).sort_values(by="Mean SHAP Value", ascending=False)

        #     print(f"Top SHAP features for {model_name}:")
        #     print(shap_importance.head(10))

        #     # Save SHAP results to CSV
        #     shap_importance.to_csv(f"{model_name}_shap_importance.csv", index=False)
        #     print(f"SHAP feature importances saved to '{model_name}_shap_importance.csv'.")

        # if model_name in ["SVM"]:
        #     print(f"Computing SHAP values for {model_name}...")
        #     explainer = shap.Explainer(best_model, X)
        #     shap_values = explainer(X)

        #     # SHAP summary plot
        #     shap.summary_plot(shap_values, X, feature_names=feats)

        #     # Optional: Save SHAP values for future analysis
        #     shap_importance = pd.DataFrame({
        #         "Feature": feats,
        #         "Mean SHAP Value": abs(shap_values.values).mean(axis=0)
        #     }).sort_values(by="Mean SHAP Value", ascending=False)

        #     print(f"Top SHAP features for {model_name}:")
        #     print(shap_importance.head(10))

            # Save SHAP results to CSV
            # shap_importance.to_csv(f"{model_name}_shap_importance.csv", index=False)
            # print(f"SHAP feature importances saved to '{model_name}_shap_importance.csv'.")

        
        # Evaluate the best model across metrics
        auc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='roc_auc')*100
        f1_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='f1_macro')*100
        accuracy_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='accuracy')*100
        balanced_acc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='balanced_accuracy')*100
        aupoc_scores = cross_val_score(best_model, X, y, groups=groups, cv=kf, scoring='average_precision')*100

        # Append evaluation results
        results["Model"].append(model_name)
        results["AUC"].append(f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
        # results["F1"].append(f"{np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
        # results["Accuracy"].append(f"{np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
        results["Balanced Accuracy"].append(f"{np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
        results["Average Precision (AUPOC)"].append(f"{np.mean(aupoc_scores):.2f} ± {np.std(aupoc_scores):.2f}")
        # results["Best Parameters"].append(best_params)

        # Save interim results
        pd.DataFrame(results).to_csv('interim_model_evaluations.csv', index=False)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            gender_test = gender_labels[test_idx]

            # Train model on this fold
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            # Evaluate per gender
            for gender in np.unique(gender_test):
                idx = np.where(gender_test == gender)[0]
                if len(idx) == 0:
                    continue
                acc = accuracy_score(y_test[idx], y_pred[idx])
                f1 = f1_score(y_test[idx], y_pred[idx], average='binary')  # or 'macro'/'weighted' if you prefer
                gender_results["Model"].append(model_name)
                gender_results["Gender"].append(gender)
                gender_results["Accuracy"].append(acc)
                gender_results["F1"].append(f1)

    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Convert final results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

gender_df = pd.DataFrame(gender_results)
print(gender_df)

#%% Save the results to a CSV file
results_df.to_csv('all_mode_outputs.csv', index=False)
# %%
from itertools import combinations

feats = ['HR_mean','HR_std','meanNN','SDNN','meanSD','SDSD','RMSSD','pNN20','pNN50','TINN','HF','SQI','SNR','SNR_freq']

# Define all possible feature combinations
max_features_in_subset = len(feats)
feature_combinations = []
for r in range(1, max_features_in_subset + 1):  # r is the size of combinations
    feature_combinations.extend(combinations(feats, r))

# Initialize results storage
results = {
    "Feature Combination": [],
    "Model": [],
    "AUC": [],
    # "F1": [],
    # "Accuracy": [],
    # "Balanced Accuracy": [],
    "Average Precision (AUPOC)": []
    # "Best Parameters": []
}

# Iterate over feature combinations
for feature_subset in feature_combinations:
    subset_name = ", ".join(feature_subset)
    X_subset = data[list(feature_subset)].values  # Select features in the subset
    X_subset = sc.fit_transform(X_subset)  # Normalize the subset

    # Perform grid search for each model
    for model_name, model in models.items():
        print(f"Training {model_name} with features: {subset_name}...")
        try:
            # Perform Grid Search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=kf,
                scoring='average_precision',
                n_jobs=2
            )
            grid_search.fit(X_subset, y, groups=groups)

            # Retrieve best parameters and model
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            # Evaluate the best model across metrics
            auc_scores = cross_val_score(best_model, X_subset, y, groups=groups, cv=kf, scoring='roc_auc')*100
            f1_scores = cross_val_score(best_model, X_subset, y, groups=groups, cv=kf, scoring='f1_macro')*100
            accuracy_scores = cross_val_score(best_model, X_subset, y, groups=groups, cv=kf, scoring='accuracy')*100
            balanced_acc_scores = cross_val_score(best_model, X_subset, y, groups=groups, cv=kf, scoring='balanced_accuracy')*100
            aupoc_scores = cross_val_score(best_model, X_subset, y, groups=groups, cv=kf, scoring='average_precision')*100

            # Append evaluation results
            results["Feature Combination"].append(subset_name)
            results["Model"].append(model_name)
            results["AUC"].append(f"{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
            # results["F1"].append(f"{np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
            # results["Accuracy"].append(f"{np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
            # results["Balanced Accuracy"].append(f"{np.mean(balanced_acc_scores):.2f} ± {np.std(balanced_acc_scores):.2f}")
            results["Average Precision (AUPOC)"].append(f"{np.mean(aupoc_scores):.2f} ± {np.std(aupoc_scores):.2f}")
            # results["Best Parameters"].append(best_params)

            # Save interim results
            pd.DataFrame(results).to_csv('interim_model_evaluations_with_features.csv', index=False)

        except Exception as e:
            print(f"Error with {model_name} and features {subset_name}: {e}")

# Convert final results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

# Save the final results to a CSV file
results_df.to_csv('final_model_evaluations_with_features.csv', index=False)

# %%
import pandas as pd

# Load the CSV file
file_path = "interim_model_evaluations_with_features.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Convert the "Average Precision (AUPOC)" column to numeric by splitting out the mean value
df["Average Precision (AUPOC)"] = df["Average Precision (AUPOC)"].str.split(" ± ").str[0].astype(float)

# Group by feature combination and calculate the mean Average Precision (AUPOC) for each combination
top_combinations = (
    df.groupby("Feature Combination")["Average Precision (AUPOC)"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

# Extract the corresponding rows for the top combinations
top_combinations_data = df[df["Feature Combination"].isin(top_combinations.index)]

# Display the results
print("Top 5 Feature Combinations by Average Precision (AUPOC):")
print(top_combinations_data)

# Save the results to a new CSV (optional)
output_file = "top_feature_combinations.csv"
top_combinations_data.to_csv(output_file, index=False)
print(f"Top combinations saved to {output_file}")

# %% look at shapley values to help select important features

