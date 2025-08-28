#%% Efficiency Measurement Script for the WESAD dataset using TD features only
import time
import sys
import os
import psutil
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import csv

# import processing functions
sys.path.append('../../src/')

from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize, simple_dynamic_threshold, simple_noise_elimination

def read_csv(path, feats, testset_num):
    print("testset num: ",testset_num)
    df = pd.read_csv(path, index_col = 0)
    
    df = df[feats]
    # no longer need the test subject separated
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

def SVM_model_with_timing(X_train, y_train, X_test, y_test):
    """
    Simple SVM model with timing measurements (no GridSearch - matches existing WESAD analysis)
    """
    # Measure training time
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Simple SVM (matching your existing approach)
    model = svm.SVC()
    model.fit(X_train, y_train)
    
    # End training timing
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    training_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    # Measure model size
    model_size_bytes = len(pickle.dumps(model))
    model_size_kb = model_size_bytes / 1024
    
    # Measure scaler size (since we need both for deployment)
    scaler_size_bytes = len(pickle.dumps(StandardScaler().fit(X_train)))
    scaler_size_kb = scaler_size_bytes / 1024
    total_deployment_size_kb = model_size_kb + scaler_size_kb
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics (same as your original function)
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy, training_time, model_size_kb, total_deployment_size_kb, model

def measure_inference_time(model, scaler, n_tests=100):
    """
    Measure inference time for a single sample using raw PPG data from CSV file
    
    # load the raw data from CSV
    # process the raw data (standardize, bandpass filter, smooth, and noise elimination)
    # extract features and make prediction with processed data
    """
    
    # Load the raw PPG data from CSV (assuming first column contains PPG values)
    try:
        raw_ppg_df = pd.read_csv(test_data_path, index_col=0)
        # Get the first column (assuming it contains PPG values)
        raw_ppg = raw_ppg_df.iloc[:, 0].values
        print(f"Loaded {len(raw_ppg)} PPG samples from {test_data_path}")
    except Exception as e:
        print(f"Error loading test data from {test_data_path}: {e}")
        return np.nan, np.nan
    
    # Check if we have enough data
    if len(raw_ppg) < 128:
        print(f"Warning: PPG signal is too short ({len(raw_ppg)} samples). Need at least 128 samples.")
        return np.nan, np.nan
    
    # Remove NaN values
    clean_ppg_values = raw_ppg[~np.isnan(raw_ppg)]
    
    if len(clean_ppg_values) < 128:
        print(f"Warning: After removing NaN values, PPG signal is too short ({len(clean_ppg_values)} samples).")
        return np.nan, np.nan
    
    inference_times = []
    successful_inferences = 0

    print(f"Running {n_tests} inference tests with {len(clean_ppg_values)} PPG samples...")
    
    for i in range(n_tests):
        start_time = time.perf_counter()

        try:
            # Process for TD features
            ppg_standardized = standardize(clean_ppg_values)
            
            # Apply filtering for TD features
            bp_bvp = bandpass_filter(ppg_standardized, 0.5, 10, 64, order=2)
            smoothed_signal = moving_average_filter(bp_bvp, window_size=5)

            # Apply noise elimination
            segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95)
            clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
            
            # Final smoothing
            final_clean_signal = moving_average_filter(clean_signal, window_size=3)
            
            # Extract TD features
            td_stats = get_ppg_features(ppg_seg=final_clean_signal.tolist(), 
                                      fs=64, 
                                      label=1, 
                                      calc_sq=True)

            # Process features if extraction was successful
            if td_stats is not None and 'sqi' in td_stats:
                td_stats['SQI'] = td_stats.pop('sqi')
                td_stats.pop('label', None)

                # Ensure feature order matches training (exclude 'subject' and 'label')
                feature_names = [feat for feat in feats if feat not in ['subject', 'label']]
                feature_vector = np.array([td_stats[feat] for feat in feature_names]).reshape(1, -1)

                # Scale features using the same scaler from training
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Make prediction
                prediction = model.predict(feature_vector_scaled)
                
                successful_inferences += 1
            else:
                print(f"Warning: Feature extraction failed for inference {i}")

        except Exception as e:
            print(f"Error in inference {i}: {e}")
            
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    if len(inference_times) == 0:
        print("No successful inferences!")
        return np.nan, np.nan
        
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"✓ Inference measurement completed! ({successful_inferences}/{n_tests} successful)")
    print(f"  Average inference time: {avg_inference_time:.3f}ms")
    print(f"  Standard deviation: {std_inference_time:.3f}ms")
    
    return avg_inference_time, std_inference_time

#%% Main execution with efficiency measurements
print("\n" + "="*50)
print("WESAD TD FEATURES - EFFICIENCY MEASUREMENT")
print("="*50)

# Data definitions
data_path = '../../data/WESAD_all_subjects_TD_features.csv'
test_data_path = '../../data/WESAD_inference_test_signal.csv'

result_path_all = '../../results/WESAD/Efficiency/TD_results.csv'
result_path_efficiency = '../../results/WESAD/Efficiency/TD_efficiency_detailed.csv'

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','subject','label']

subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

# Storage for results
SVM_AUC, SVM_F1, SVM_ACC = [], [], []
training_times, model_sizes, deployment_sizes = [], [], []
avg_inference_times, std_inference_times = [], []

print(f"Running LOGO validation for {len(subjects)} subjects...")

for i, sub in enumerate(subjects):
    print(f"\n--- Subject {sub} ({i+1}/{len(subjects)}) ---")
    
    df, X_train, y_train, X_test, y_test = read_csv(data_path, feats, sub)
    df = df.fillna(0)  # Fixed: was missing assignment
    
    # Normalization
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    # Train with efficiency measurements
    auc_svm, f1_svm, acc_svm, train_time, model_size_kb, deploy_size_kb, model = SVM_model_with_timing(
        X_train, y_train, X_test, y_test
    )
    
    # Measure inference time using raw PPG data
    avg_inf_time, std_inf_time = measure_inference_time(model, sc)
    
    # Store results
    SVM_AUC.append(auc_svm*100)
    SVM_F1.append(f1_svm*100)
    SVM_ACC.append(acc_svm*100)
    training_times.append(train_time)
    model_sizes.append(model_size_kb)
    deployment_sizes.append(deploy_size_kb)
    avg_inference_times.append(avg_inf_time)
    std_inference_times.append(std_inf_time)
    
    print(f"  AUC: {auc_svm*100:.2f}%, F1: {f1_svm*100:.2f}%, Acc: {acc_svm*100:.2f}%")
    print(f"  Train time: {train_time:.3f}s, Model: {model_size_kb:.2f}KB, Inference: {avg_inf_time:.3f}ms")

# Calculate summary statistics (handle NaN values)
valid_inference_times = [t for t in avg_inference_times if not np.isnan(t)]
valid_std_times = [t for t in std_inference_times if not np.isnan(t)]

print(f"\n" + "="*80)
print("WESAD TD FEATURES - SUMMARY RESULTS")
print("="*80)

print("\nPerformance Metrics:")
print("-" * 40)
print(f"AUC-ROC: {np.mean(SVM_AUC):.2f} ± {np.std(SVM_AUC):.2f}")
print(f"F1 Score: {np.mean(SVM_F1):.2f} ± {np.std(SVM_F1):.2f}")
print(f"Accuracy: {np.mean(SVM_ACC):.2f} ± {np.std(SVM_ACC):.2f}")

print("\nEfficiency Metrics:")
print("-" * 40)
print(f"Avg Training Time: {np.mean(training_times):.3f} ± {np.std(training_times):.3f} seconds")
print(f"Avg Model Size: {np.mean(model_sizes):.2f} ± {np.std(model_sizes):.2f} KB")
print(f"Avg Deployment Size: {np.mean(deployment_sizes):.2f} ± {np.std(deployment_sizes):.2f} KB")

if valid_inference_times:
    print(f"Avg Inference Time: {np.mean(valid_inference_times):.3f} ± {np.std(valid_inference_times):.3f} ms")
else:
    print("Avg Inference Time: No valid measurements")

# Create results directory if it doesn't exist
os.makedirs('../../results/WESAD/Efficiency', exist_ok=True)

# Save efficiency details
efficiency_results = pd.DataFrame({
    'subject': subjects,
    'auc_roc': SVM_AUC,
    'f1_score': SVM_F1,
    'accuracy': SVM_ACC,
    'training_time_s': training_times,
    'model_size_KB': model_sizes,
    'deployment_size_KB': deployment_sizes,
    'avg_inference_time_ms': avg_inference_times,
    'std_inference_time_ms': std_inference_times
})

efficiency_results.to_csv(result_path_efficiency, index=False)

# Save summary for comparison with other approaches
summary_results = {
    'approach': 'TD_features_only',
    'mean_training_time_s': np.mean(training_times),
    'std_training_time_s': np.std(training_times),
    'mean_model_size_KB': np.mean(model_sizes),
    'std_model_size_KB': np.std(model_sizes),
    'mean_deployment_size_KB': np.mean(deployment_sizes),
    'std_deployment_size_KB': np.std(deployment_sizes),
    'mean_inference_time_ms': np.mean(valid_inference_times) if valid_inference_times else np.nan,
    'std_inference_time_ms': np.std(valid_inference_times) if valid_inference_times else np.nan
}

summary_df = pd.DataFrame([summary_results])
summary_df.to_csv(result_path_all, index=False)

print(f"\nResults saved to:")
print(f"  - {result_path_all}")
print(f"  - {result_path_efficiency}")
# %%