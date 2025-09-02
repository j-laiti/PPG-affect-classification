#%% Efficiency Measurement Script for the AKTIVES dataset using the proposed pipeline
import time
import sys
import os
import psutil
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Load TD features data
data = pd.read_csv('../../data/Aktives/extracted_features/ppg_features_combined.csv')

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['participant'].nunique()}")
print(f"Label distribution:\n{data['label'].value_counts()}")

#%% Feature selection
feats = ['HR_mean','HR_std','SDNN','SDSD','meanNN','medianNN','RMSSD','meanSD','sqi'] #no pnn50/20 used because of data length
label = 'label'

X = data[feats].values
y = data[label].values

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

#%% 70/30 split from the training data (excluding inference sample)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_for_training, test_size=0.3, random_state=42, stratify=y_for_training
)

print(f"\nTraining split:")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Train label distribution: {np.bincount(y_train)}")
print(f"Test label distribution: {np.bincount(y_test)}")

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
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # time *training* as fit() only
    final_model = svm.SVC(probability=True, class_weight='balanced', random_state=42, **best_params)
    t0 = time.perf_counter()
    final_model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0
    print(f"✓ Final fit() training time: {training_time:.3f}s")
    
    # End memory calculation
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

    # After the model size measurement, add:
    scaler_size_bytes = len(pickle.dumps(scaler))
    scaler_size_kb = scaler_size_bytes / 1024
    total_deployment_size_kb = model_size_kb + scaler_size_kb

    print(f"  Scaler size: {scaler_size_bytes} bytes ({scaler_size_kb:.2f} KB)")
    print(f"  Total deployment size: {total_deployment_size_kb:.2f} KB")
    
except Exception as e:
    print(f"Error during training: {e}")






#%% Measure Complete Inference Time (Feature Extraction + Prediction)
print("\n" + "="*50)
print("MEASURING COMPLETE INFERENCE TIME (Raw PPG → Features → Prediction)")
print("="*50)

# We need to get the raw data info for our inference sample
# First, let's get the metadata for our reserved sample
inference_sample_metadata = data.iloc[test_sample_idx]

print(f"Inference sample metadata:")
print(f"  Participant: {inference_sample_metadata['participant']}")
print(f"  Game: {inference_sample_metadata['game']}")  
print(f"  Cohort: {inference_sample_metadata['cohort']}")
print(f"  Label: {inference_sample_metadata['label']}")

# Import your feature extraction functions
sys.path.append('../..')

from preprocessing.feature_extraction import *
from preprocessing.filters import *

# Modified feature extraction function
def extract_td_features_for_inference_sample(sample_metadata, fs=64):
    """Extract TD features for one sample"""
    try:
        participant = sample_metadata['participant']
        game = sample_metadata['game']
        cohort = sample_metadata['cohort']
        interval_start = sample_metadata['interval_start']
        interval_end = sample_metadata['interval_end']
        label = int(sample_metadata['label'])
        
        # Map cohort names to folder names
        cohort_folder_map = {
            'dyslexia': 'Dyslexia',
            'ID': 'Intellectual Disabilities', 
            'OBPI': 'Obstetric Brachial Plexus Injuries',
            'TD': 'Typically Developed'
        }
        
        cohort_folder = cohort_folder_map[cohort]
        ppg_file_path = f"../../data/Aktives/PPG/{cohort_folder}/{participant}/{game}/BVP.csv"
        
        if not os.path.exists(ppg_file_path):
            return None
            
        # Load PPG data
        ppg_data = pd.read_csv(ppg_file_path)
        ppg_data["Time"] = ppg_data.index / fs
        ppg_data['values'] = ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # Select interval
        ppg_interval = ppg_data[(ppg_data['Time'] >= interval_start) & 
                               (ppg_data['Time'] <= interval_end)]
        raw_ppg_values = ppg_interval['values'].values
        
        # Check for sufficient data
        if len(raw_ppg_values) < fs * 10:  # Less than 10 seconds
            return None
        
        # ===== TIME-DOMAIN FEATURE EXTRACTION =====
        td_extraction_time_start = time.perf_counter()
        # Process for TD features (same as before)
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        ppg_standardized = standardize(clean_ppg_values)
        
        # Apply filtering for TD features
        bp_bvp = bandpass_filter(ppg_standardized, 0.5, 10, fs, order=2)
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)

        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95, window_size= 3)
        sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
        sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
        
        # Extract TD features
        td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                  fs=fs, 
                                  label=label, 
                                  calc_sq=True)
        
        # print(td_stats)

        # Exclude the pnn50 and pnn20 from the TD features
        if td_stats is not None:
            td_stats.pop('pNN50', None)
            td_stats.pop('pNN20', None)
            td_stats.pop('label', None)

        if td_stats is None:
            return None
        
        td_extraction_time_end = time.perf_counter()
        td_extraction_time = td_extraction_time_end - td_extraction_time_start
            
        # Convert TD stats to list
        if isinstance(td_stats, dict):
            td_features = list(td_stats.values())
        else:
            td_features = td_stats
        
        # Combine CNN and TD features
        return td_features, td_extraction_time
        
    except Exception as e:
        print(f"Error extracting hybrid features: {e}")
        return None

#%% Test and measure complete hybrid inference
print("Testing proposed pipeline for inference sample...")
    
# Now measure complete TD inference time
n_inference_tests = 10
inference_times = []

print(f"Running {n_inference_tests} TD inference tests...")

for i in range(n_inference_tests):

    # Step 1: Extract TD features from raw PPG
    td_features, td_extraction_time = extract_td_features_for_inference_sample(
        inference_sample_metadata
    )

    feature_prediction_time_start = time.perf_counter()

    if td_features is not None:
        
        # Step 3: Combine all features (CNN + TD)
        feature_vector = np.array(td_features).reshape(1, -1)

        # Step 4: Scale features using the same scaler from training
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Step 5: Make prediction
        prediction = final_model.predict(feature_vector_scaled)

    feature_prediction_time_end = time.perf_counter()
    feature_prediction_time = feature_prediction_time_end - feature_prediction_time_start
    inference_times.append((td_extraction_time - feature_prediction_time) * 1000)  # Convert to milliseconds

avg_inference_time = np.mean(inference_times)
std_inference_time = np.std(inference_times)

print(f"✓ Complete hybrid inference measurement completed!")
print(f"  Average complete inference time: {avg_inference_time:.3f}ms")
print(f"  Standard deviation: {std_inference_time:.3f}ms")
print(f"  Final prediction: {prediction[0]} (actual: {inference_sample_metadata['label']})")

# %% Save results

results = {
    "training_time_s": training_time,
    "model_size_KB": model_size_kb,
    "total_deployment_size_KB": total_deployment_size_kb,
    "average_inference_time_ms": avg_inference_time,
    "std_inference_time_ms": std_inference_time,
}


output_csv_path = "AKTIVES_TD_results.csv"

save_df = pd.DataFrame([results])
save_df.to_csv(output_csv_path, index=False)

# %%
