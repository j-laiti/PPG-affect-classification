#%% Efficiency Measurement Script for the Wellby dataset using the proposed pipeline
import time
import os
import psutil
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys

# Configuration
# File paths
DATA_PATH = '../../data/Wellby/Wellby_all_subjects_features.csv'

label = 'stress_binary'
feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','RMSSD','pNN20','pNN50','PSS','PSQI','EPOCH','SQI']

# Load data
data = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['Participant'].nunique()}")
print(f"Label distribution:\n{data[label].value_counts()}")

# Separate features and labels
X = data[feats].values
y = data[label].values
groups = data['Participant'].values

print(f"Total features: {X.shape[1]}")

#%% Prepare data splits - reserve one sample for inference testing
np.random.seed(42)
test_sample_idx = np.random.choice(len(X), 1)[0]

# Get the test sample for inference timing
X_inference_sample = X[test_sample_idx:test_sample_idx+1]
y_inference_sample = y[test_sample_idx]

# Remove test sample from training data
X_for_training = np.delete(X, test_sample_idx, axis=0)
y_for_training = np.delete(y, test_sample_idx)

print(f"Reserved sample for inference testing: index {test_sample_idx}, label {y_inference_sample}")
print(f"Training data shape: {X_for_training.shape}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_for_training)
X_inference_scaled = scaler.transform(X_inference_sample)

# group k-fold cross-validation
kf = StratifiedGroupKFold(n_splits=3)

# SVM Training with Timing
print("\n" + "="*50)
print("MEASURING SVM TRAINING TIME")
print("="*50)

# Parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5, 1.0]
}

# Initialize model
svm_model = svm.SVC(probability=True, class_weight='balanced', random_state=42)

# Measure training time
print("Starting SVM training with GridSearch...")
start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

try:
    # Grid search with cross-validation
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    grid_search = GridSearchCV(
        estimator=svm_model, 
        param_grid=param_grid, 
        cv=inner_cv, 
        scoring='roc_auc', 
        n_jobs=1
    )
    grid_search.fit(X_scaled, y_for_training)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # time *training* as fit() only
    final_model = svm.SVC(probability=True, class_weight='balanced', random_state=42, **best_params)
    t0 = time.perf_counter()
    final_model.fit(X_scaled, y_for_training)
    training_time = time.perf_counter() - t0
    print(f"✓ Final fit() training time: {training_time:.3f}s")
    
    # End memory measure
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_used = end_memory - start_memory
    
    print(f"✓ SVM training completed!")
    print(f"  Training time: {training_time:.3f} seconds")
    print(f"  Memory used during training: {memory_used:.1f} MB")
    print(f"  Best parameters: {best_params}")
    print(f"  Best CV score: {grid_search.best_score_:.3f}")

    # Measure model size
    model_size_bytes = len(pickle.dumps(final_model))
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024

    print(f"  Model size: {model_size_bytes} bytes ({model_size_kb:.2f} KB, {model_size_mb:.4f} MB)")

    # Measure scaler size
    scaler_size_bytes = len(pickle.dumps(scaler))
    scaler_size_kb = scaler_size_bytes / 1024
    total_deployment_size_kb = model_size_kb + scaler_size_kb

    print(f"  Scaler size: {scaler_size_bytes} bytes ({scaler_size_kb:.2f} KB)")
    print(f"  Total deployment size: {total_deployment_size_kb:.2f} KB")
    
except Exception as e:
    print(f"Error during training: {e}")




#%% Measure inference time (Feature Scaling + Prediction)
print("\n" + "="*50)
print("MEASURING INFERENCE TIME (Pre-extracted Features → Prediction)")
print("="*50)

# import processing functions
sys.path.append('../..')

from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize, simple_dynamic_threshold, simple_noise_elimination

# Get the metadata for our inference sample
inference_sample_metadata = data.iloc[test_sample_idx]

#%%
print(f"Inference sample metadata:")
print(f"  Participant: {inference_sample_metadata['Participant']}")
print(f"  Session ID: {inference_sample_metadata['Session_ID']}")
# extract session ID
session_id = inference_sample_metadata['Session_ID']

# load raw data
raw_ppg = pd.read_csv('../../data/Wellby/selected_ppg_data.csv')

# find the raw data for that session ID
test_data = raw_ppg[session_id]

#%% Test inference with the reserved sample
print("Testing inference for reserved sample...")

if test_data is not None:

    # Now measure inference time
    n_inference_tests = 100
    inference_times = []
    
    print(f"Running {n_inference_tests} inference tests...")
    
    for i in range(n_inference_tests):
        start_time = time.perf_counter()

        # go through processing pipeline to extract the necessary TD features
                # Process for TD features (same as before)
        clean_ppg_values = test_data[~np.isnan(test_data)]
        ppg_standardized = standardize(clean_ppg_values)
        
        # Apply filtering for TD features
        bp_bvp = bandpass_filter(ppg_standardized, 0.5, 10, 50, order=2)
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)

        # Apply noise elimination
        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 50, 95)
        clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, 50, std_ths)
        
        # Final smoothing
        final_clean_signal = moving_average_filter(clean_signal, window_size=3)
        
        # Extract TD features
        td_stats = get_ppg_features(ppg_seg=final_clean_signal.tolist(), 
                                  fs=50, 
                                  label=label, 
                                  calc_sq=True)

        # remove label and add survey data
        if td_stats is not None and 'sqi' in td_stats:
            td_stats['SQI'] = td_stats.pop('sqi')
            td_stats.pop('label', None)
            td_stats['PSS'] = inference_sample_metadata['PSS']
            td_stats['PSQI'] = inference_sample_metadata['PSQI']
            td_stats['EPOCH'] = inference_sample_metadata['EPOCH']

            # Ensure feature order matches training
            feature_vector = np.array([td_stats[feat] for feat in feats]).reshape(1, -1)

            # scale features using the same scaler from training
            feature_vector_scaled = scaler.transform(feature_vector)
            # make prediction
            prediction = final_model.predict(feature_vector_scaled)

        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"✓ Complete inference measurement completed!")
    print(f"  Average inference time: {avg_inference_time:.3f}ms")
    print(f"  Standard deviation: {std_inference_time:.3f}ms")
    print(f"  Final prediction: {prediction[0]} (actual: {inference_sample_metadata[label]})")

    # Save results
    results = {
        "training_time_s": training_time,
        "model_size_KB": model_size_kb,
        "total_deployment_size_KB": total_deployment_size_kb,
        "average_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
    }

    # Create results directory if it doesn't exist
    
    output_csv_path = "Wellby_TD_inference.csv"
    
    save_df = pd.DataFrame([results])
    save_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ Results saved to: {output_csv_path}")

else:
    print("❌ Could not retrieve features for inference sample")
# %%
