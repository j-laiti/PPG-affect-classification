# Wellby efficiency calculation using the 3-fold cross validation with the CNN + TD combined feature set

#%% imports
import time
import os
import psutil
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys

# Add path for preprocessing functions
sys.path.append('../../src/')
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize

# Configuration
# File paths
DATA_PATH = '../../data/Wellby_hybrid_features/wellby_stress_hybrid_features.csv'
RAW_PPG_PATH = '../../data/Wellby/selected_ppg_data.csv'
RESULTS_DIR = '../../results/Wellby/Efficiency/'

TASK = 'stress'
output_filename = 'hybrid_results.csv'

# Define the CNN architecture for feature extraction (matching your training code)
class Simple1DCNNFeatures(nn.Module):
    """
    Simple CNN for feature extraction from PPG signals
    """
    def __init__(self, input_length=3000, output_features=32):  # Adapted for ~1 min at 50Hz
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection (adapted for 50Hz vs 64Hz)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Layer 2: Pattern refinement
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Layer 3: High-level features
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, output_features//4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # 32 features output (8*4=32)
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

def prepare_ppg_for_cnn(ppg_signal, target_length=3000, skip_start_samples=100, fs=50):
    """
    Prepare raw PPG signal for CNN feature extraction
    """
    # Skip initial noisy samples
    if len(ppg_signal) <= skip_start_samples:
        return None
        
    trimmed_signal = ppg_signal[skip_start_samples:]
    
    # Handle variable length - truncate to target_length
    if len(trimmed_signal) >= target_length:
        processed_signal = trimmed_signal[:target_length]
    else:
        # Zero pad if shorter
        padding = target_length - len(trimmed_signal)
        processed_signal = np.pad(trimmed_signal, (0, padding), mode='constant', constant_values=0)
    
    # Remove NaN values
    clean_signal = processed_signal[~np.isnan(processed_signal)]
    if len(clean_signal) == 0:
        return None
    
    # Basic standardization
    standardized_signal = standardize(clean_signal)
    
    # Normalize for CNN
    normalized_signal = (standardized_signal - np.mean(standardized_signal)) / (np.std(standardized_signal) + 1e-8)
    
    return normalized_signal

def extract_td_features_from_ppg(ppg_signal, skip_start_samples=100, fs=50):
    """
    Extract time-domain features from raw PPG signal
    """
    # Skip initial noisy samples
    if len(ppg_signal) <= skip_start_samples:
        return None
        
    trimmed_signal = ppg_signal[skip_start_samples:]
    
    # Remove NaN values
    clean_signal = trimmed_signal[~np.isnan(trimmed_signal)]
    if len(clean_signal) == 0:
        return None
    
    # Standardize
    standardized_signal = standardize(clean_signal)
    
    # Apply filtering for TD features (following WESAD approach)
    bp_signal = bandpass_filter(standardized_signal, 0.5, 10, fs, order=2)
    smoothed_signal = moving_average_filter(bp_signal, window_size=5)
    
    # Extract TD features
    td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                              fs=fs, 
                              label=0,  # Dummy label for feature extraction
                              calc_sq=False)
    
    if td_stats is None:
        return None
    
    # Convert to array format
    if isinstance(td_stats, dict):
        # Remove label if present
        td_stats.pop('label', None)
        td_features = list(td_stats.values())
    else:
        td_features = td_stats
    
    return td_features

#%% Feature selection - Combined CNN + TD features
# Load hybrid features data
data = pd.read_csv(DATA_PATH)

# Load raw PPG data for inference testing
raw_ppg_data = pd.read_csv(RAW_PPG_PATH)

print(f"Dataset shape: {data.shape}")
print(f"Raw PPG data shape: {raw_ppg_data.shape}")

# Identify CNN features
cnn_features = [col for col in data.columns if col.startswith('cnn_feature_')]
print(f"CNN features found: {len(cnn_features)}")

# Identify TD features
metadata_cols = ['session_id','participant', 'school', 'age', 'gender', 'label', 'PSS', 'PSQI', 'EPOCH', 'SQI']
td_features = [col for col in data.columns if col not in cnn_features and col not in metadata_cols]
print(f"TD features found: {len(td_features)}")
print(f"TD feature names: {td_features}")

# Combine all features
all_features = cnn_features + td_features
label = 'label'

X = data[all_features].values
y = data[label].values

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
    
    # End timing
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
    
    # Create a dummy CNN model for size estimation
    device = torch.device('cpu')  # Use CPU for inference timing
    dummy_cnn = Simple1DCNNFeatures(input_length=3000, output_features=32)
    
    # Save and measure CNN model size
    torch.save(dummy_cnn.state_dict(), 'temp_cnn_model.pth')
    cnn_size_bytes = os.path.getsize('temp_cnn_model.pth')
    cnn_size_kb = cnn_size_bytes / 1024
    os.remove('temp_cnn_model.pth')  # Clean up
    
    total_deployment_size_kb = model_size_kb + scaler_size_kb + cnn_size_kb

    print(f"  Scaler size: {scaler_size_bytes} bytes ({scaler_size_kb:.2f} KB)")
    print(f"  CNN model size: {cnn_size_bytes} bytes ({cnn_size_kb:.2f} KB)")
    print(f"  Total deployment size: {total_deployment_size_kb:.2f} KB")
    
except Exception as e:
    print(f"Error during training: {e}")

#%% Run inference testing on the reserved sample

print("\n" + "="*50)
print("MEASURING COMPLETE HYBRID INFERENCE TIME (Raw PPG → CNN + TD Features → Prediction)")
print("="*50)

# Get the metadata for our inference sample
inference_sample_metadata = data.iloc[test_sample_idx]

print(f"Inference sample metadata:")
print(f"  Participant: {inference_sample_metadata['participant']}")
print(f"  Session ID: {inference_sample_metadata['session_id']}")

# Extract session ID
session_id = inference_sample_metadata['session_id']

# Check if session ID exists in raw PPG data
if session_id in raw_ppg_data.columns:
    # Get raw PPG data for this session
    test_ppg_signal = raw_ppg_data[session_id].dropna().values
    print(f"✓ Found raw PPG signal with {len(test_ppg_signal)} samples")
    
    # Initialize CNN model for feature extraction
    device = torch.device('cpu')  # Use CPU for realistic edge inference
    cnn_model = Simple1DCNNFeatures(input_length=3000, output_features=32).to(device)
    cnn_model.eval()
    
    # Test inference with the reserved sample
    print("Testing hybrid inference for reserved sample...")
    
    # Now measure complete hybrid inference time
    n_inference_tests = 100
    inference_times = []
    
    print(f"Running {n_inference_tests} hybrid inference tests...")
    
    successful_inferences = 0
    
    for i in range(n_inference_tests):
        start_time = time.perf_counter()
        
        try:
            # Step 1: Prepare PPG signal for CNN
            cnn_signal = prepare_ppg_for_cnn(test_ppg_signal, target_length=3000, skip_start_samples=100)
            
            # Step 2: Extract CNN features
            if cnn_signal is not None:
                with torch.no_grad():
                    signal_tensor = torch.FloatTensor(cnn_signal).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, length)
                    cnn_features = cnn_model(signal_tensor).cpu().numpy().flatten()
            else:
                cnn_features = np.zeros(32)  # Fallback if signal processing fails
            
            # Step 3: Extract TD features
            td_features = extract_td_features_from_ppg(test_ppg_signal, skip_start_samples=100)
            if td_features is None:
                td_features = np.zeros(len(td_features))  # Fallback if TD extraction fails
            
            # Step 4: Combine all features
            combined_features = np.concatenate([cnn_features, td_features]).reshape(1, -1)
            
            # Step 5: Scale features using the same scaler from training
            combined_features_scaled = scaler.transform(combined_features)
            
            # Step 6: Make prediction
            prediction = final_model.predict(combined_features_scaled)

            successful_inferences += 1
            
        except Exception as e:
            print(f"  Warning: Inference {i+1} failed: {e}")
            # Use fallback values for failed inference
            prediction = [0]  # Default prediction
        
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"✓ Complete hybrid inference measurement completed!")
    print(f"  Successful inferences: {successful_inferences}/{n_inference_tests}")
    print(f"  Average complete inference time: {avg_inference_time:.3f}ms")
    print(f"  Standard deviation: {std_inference_time:.3f}ms")
    print(f"  Final prediction: {prediction[0]} (actual: {inference_sample_metadata['label']})")

    # Save results
    results = {
        "training_time_s": training_time,
        "model_size_KB": model_size_kb,
        "cnn_model_size_KB": cnn_size_kb,
        "scaler_size_KB": scaler_size_kb,
        "total_deployment_size_KB": total_deployment_size_kb,
        "average_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "successful_inferences": successful_inferences,
        "total_features": len(all_features),
        "cnn_features": len(cnn_features),
        "td_features": len(td_features)
    }

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    output_csv_path = f"{RESULTS_DIR}/{output_filename}"
    
    save_df = pd.DataFrame([results])
    save_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ Results saved to: {output_csv_path}")

else:
    print(f"❌ Could not find session ID '{session_id}' in raw PPG data")
    print(f"Available session IDs: {raw_ppg_data.columns[:10].tolist()}...")  # Show first 10
# %%
