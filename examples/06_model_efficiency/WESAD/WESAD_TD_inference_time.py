"""
WESAD Dataset TD Efficiency Inference Time Evaluation
Table X in the paper
DOI: 10.1109/TAFFC.2025.3628467

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% imports
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
sys.path.append('../..')

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
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Simple SVM (matching your existing approach)
    model = svm.SVC()
    model.fit(X_train, y_train)
    
    # End training timing
    end_time = time.perf_counter()
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

def extract_td_features_for_inference_sample(subject_id, fs=64):
    """Extract TD features for inference timing using WESAD dataset structure (CONSISTENT WITH AKTIVES)"""
    
    # Configuration - WESAD specific
    WINDOW_SAMPLES = 120 * 64  # 120s at 64Hz (same as training)
    DATA_PATH = '../../data/WESAD_BVP_extracted/'
    
    try:
        # Load subject data (similar to AKTIVES approach but using WESAD structure)
        subject_file = os.path.join(DATA_PATH, f'S{subject_id}.csv')
        if not os.path.exists(subject_file):
            print(f"Subject file not found: {subject_file}")
            return None
        
        df = pd.read_csv(subject_file)
        
        # Get a sample window from the subject's data (take first available window)
        # Try stress data first, then non-stress
        for label_value in [1.0, 0.0]:
            label_data = df[df['label'] == label_value]['BVP'].values
            
            if len(label_data) >= WINDOW_SAMPLES:
                # Take the first window
                window = label_data[:WINDOW_SAMPLES]

                # ===== TIME-DOMAIN FEATURE EXTRACTION =====
                td_extraction_time_start = time.perf_counter()
                # Process for TD features (same as before)
                clean_ppg_values = window[~np.isnan(window)]
                ppg_standardized = standardize(clean_ppg_values)
                
                # Apply preprocessing pipeline (same as TD approach)
                bp_bvp = bandpass_filter(ppg_standardized, 0.2, 10, 64, order=2)
                smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
                
                segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95, window_size=3)
                sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
                sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
                
                td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                          fs=64, 
                                          label=int(label_value), 
                                          calc_sq=True)
                
                if td_stats is None or len(td_stats) <= 1:
                    print("Warning: TD feature extraction failed")
                    continue
                
                # Process features to match training format
                if 'sqi' in td_stats:
                    td_stats['SQI'] = td_stats.pop('sqi')
                td_stats.pop('label', None)
                
                td_extraction_time_end = time.perf_counter()
                td_extraction_time = td_extraction_time_end - td_extraction_time_start
                
                # Return processed features
                return td_stats, td_extraction_time
        
        print(f"No suitable window found for subject S{subject_id}")
        return None
        
    except Exception as e:
        print(f"Error extracting TD features for S{subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def measure_inference_time(subject_id, model, scaler, feats, n_tests=100):
    """
    Measure inference time using actual WESAD subject data (CONSISTENT WITH AKTIVES)
    """
    print(f"   Measuring TD inference time with subject S{subject_id} data...")
    
    inference_times = []
    successful_inferences = 0

    print(f"Running {n_tests} inference tests with subject S{subject_id}...")
    
    for i in range(n_tests):

        try:
            # Extract TD features from actual subject data
            td_stats, td_extraction_time = extract_td_features_for_inference_sample(subject_id)
            
            if td_stats is not None:
                feature_prediction_time_start = time.perf_counter()
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
            
        feature_prediction_time_end = time.perf_counter()
        feature_prediction_time = feature_prediction_time_end - feature_prediction_time_start
        inference_times.append((feature_prediction_time + td_extraction_time) * 1000)  # Convert to milliseconds
    
    if len(inference_times) == 0:
        print("No successful inferences!")
        return np.nan, np.nan
        
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"✓ Inference measurement completed! ({successful_inferences}/{n_tests} successful)")
    print(f"  Average inference time: {avg_inference_time:.3f}ms")
    print(f"  Standard deviation: {std_inference_time:.3f}ms")
    
    return avg_inference_time, std_inference_time

#%% Main execution with efficiency measurements - CONSISTENT WITH AKTIVES
print("\n" + "="*50)
print("WESAD TD FEATURES - EFFICIENCY MEASUREMENT (CONSISTENT WITH AKTIVES)")
print("="*50)

# Data definitions
data_path = '../../data/WESAD_all_subjects_TD_features.csv'

result_path_all = 'WESAD_TD_results.csv'
result_path_efficiency = 'WESAD_TD_efficiency_detailed.csv'

feats = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50','subject','label']

subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

# Storage for results
SVM_AUC, SVM_F1, SVM_ACC = [], [], []
training_times, model_sizes, deployment_sizes = [], [], []
avg_inference_times, std_inference_times = [], []

print(f"Running LOGO validation for {len(subjects)} subjects...")

for i, sub in enumerate(subjects):
    print(f"\n{'='*40}")
    print(f"Subject S{sub} ({i+1}/{len(subjects)})")
    print(f"{'='*40}")
    
    try:
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
        
        # Measure inference time using actual subject data (CONSISTENT WITH AKTIVES)
        print("Measuring TD inference time with actual subject data...")
        avg_inf_time, std_inf_time = measure_inference_time(sub, model, sc, feats)
        
        # Store results
        SVM_AUC.append(auc_svm*100)
        SVM_F1.append(f1_svm*100)
        SVM_ACC.append(acc_svm*100)
        training_times.append(train_time)
        model_sizes.append(model_size_kb)
        deployment_sizes.append(deploy_size_kb)
        avg_inference_times.append(avg_inf_time)
        std_inference_times.append(std_inf_time)
        
        print(f"Results for S{sub}:")
        print(f"  AUC: {auc_svm*100:.2f}%, F1: {f1_svm*100:.2f}%, Acc: {acc_svm*100:.2f}%")
        print(f"  Train time: {train_time:.3f}s, Model: {model_size_kb:.2f}KB, Inference: {avg_inf_time:.3f}ms")
        
    except Exception as e:
        print(f"Error processing subject S{sub}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

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

# Save efficiency details
efficiency_results = pd.DataFrame({
    'subject': subjects,
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