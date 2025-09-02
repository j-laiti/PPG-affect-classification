#%% Efficiency Measurement Script for AKTIVES using a hybrid CNN and TD feature extraction pipeline
import time
import sys
import os
import psutil
import torch
import numpy as np
import pandas as pd
import joblib  # Fixed import
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score  # Added missing imports

# Load hybrid features data
data = pd.read_csv('../../data/AKTIVES_hybrid_features/all_subjects_AKTIVES_hybrid_features_timing.csv')

print(f"Dataset shape: {data.shape}")
print(f"Unique participants: {data['participant'].nunique()}")
print(f"Label distribution:\n{data['label'].value_counts()}")

# Feature selection - Combined CNN + TD features
# Identify CNN features
cnn_features = [col for col in data.columns if col.startswith('cnn_feature_')]
print(f"CNN features found: {len(cnn_features)}")

# Identify TD features
metadata_cols = ['participant', 'game', 'cohort', 'label', 'interval_start', 'interval_end', 'pNN50', 'pNN20'] # remove pNN20 and pNN50 because of short window size
td_features = [col for col in data.columns if col not in cnn_features and col not in metadata_cols]
print(f"TD features found: {len(td_features)}")

# Combine all features
all_features = cnn_features + td_features

# Combine CNN + TD features with cohort dummies
X = data[all_features].values
y = data['label'].values

print(f"Total features: {X.shape[1]}")

#%% Check for and handle NaN values
print("\n" + "="*50)
print("CHECKING FOR NaN VALUES")
print("="*50)

# Check for NaN values in the feature matrix
print(f"Original dataset shape: {X.shape}")
print(f"NaN values per column:")

nan_counts = pd.DataFrame(data[all_features]).isnull().sum()
nan_columns = nan_counts[nan_counts > 0]

if len(nan_columns) > 0:
    print("Columns with NaN values:")
    for col, count in nan_columns.items():
        percentage = (count / len(data)) * 100
        print(f"  {col}: {count} NaNs ({percentage:.1f}%)")
    
    # Check total rows with any NaN
    rows_with_nan = pd.DataFrame(data[all_features]).isnull().any(axis=1).sum()
    print(f"\nTotal rows with any NaN values: {rows_with_nan} ({(rows_with_nan/len(data))*100:.1f}%)")
    
    # Option 1: Remove rows with any NaN values (recommended for small datasets)
    print(f"\nHandling NaN values by removing rows with any NaN...")
    
    # Create mask for rows without NaN
    valid_rows_mask = ~pd.DataFrame(data[all_features]).isnull().any(axis=1)
    
    # Apply mask to data
    X_clean = X[valid_rows_mask]
    y_clean = y[valid_rows_mask]
    data_clean = data[valid_rows_mask].reset_index(drop=True)
    
    print(f"After removing NaN rows:")
    print(f"  Clean dataset shape: {X_clean.shape}")
    print(f"  Removed {len(X) - len(X_clean)} rows")
    print(f"  Label distribution after cleaning: {np.bincount(y_clean)}")
    
    # Update variables
    X = X_clean
    y = y_clean
    data = data_clean
    
    # Verify no NaN values remain
    remaining_nans = np.isnan(X).sum()
    print(f"  Remaining NaN values in X: {remaining_nans}")
    
    if remaining_nans > 0:
        print("  ERROR: NaN values still present after cleaning!")
        
else:
    print("No NaN values found in the dataset.")

print(f"Final dataset shape for training: {X.shape}")

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
    X_scaled, y_for_training, test_size=0.3, random_state=42, stratify=y_for_training  # Fixed: use y_for_training
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

#%% Import CNN model classes (add this near the top)
import torch
import torch.nn as nn

# Add your CNN classes (copy from your feature extraction file)
class Simple1DCNNFeatures(nn.Module):
    """Simple CNN for feature extraction from PPG signals"""
    def __init__(self, input_length=1920, output_features=32):
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection
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
            nn.AdaptiveAvgPool1d(4)
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

class TrainableCNNFeatureExtractor(nn.Module):
    """CNN + classification head for training"""
    def __init__(self, input_length=1920, num_classes=2):
        super().__init__()
        self.feature_extractor = Simple1DCNNFeatures(input_length)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        if return_features:
            return outputs, features
        return outputs
    
    def extract_features_only(self, x):
        """Extract only features (for hybrid approach)"""
        return self.feature_extractor(x)

#%% Load pre-trained CNN model
print("Loading pre-trained CNN model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    cnn_model = TrainableCNNFeatureExtractor(input_length=1920).to(device)
    
    # Load the saved model weights
    cnn_model.load_state_dict(torch.load('temp_best_aktives_cnn.pth', map_location=device))
    cnn_model.eval()  # Set to evaluation mode
    
    print("CNN model loaded successfully")
    
    # Test that it works by checking the number of parameters
    total_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"Could not load CNN model: {e}")
    cnn_model = None

# Modified feature extraction function with CNN
def extract_hybrid_features_for_inference_sample(sample_metadata, cnn_model, device, fs=64, target_length=1920):
    """Extract both CNN and TD features for one sample"""
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
        
        start_time = time.perf_counter()
        
        # ===== CNN FEATURE EXTRACTION =====
        # Minimal preprocessing for CNN (same as your existing approach)
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        if len(clean_ppg_values) == 0:
            return None
            
        ppg_standardized = standardize(clean_ppg_values)
        
        # Handle variable length windows - pad or truncate to target_length
        if len(ppg_standardized) > target_length:
            processed_signal = ppg_standardized[:target_length]
        elif len(ppg_standardized) < target_length:
            padding = target_length - len(ppg_standardized)
            processed_signal = np.pad(ppg_standardized, (0, padding), mode='constant', constant_values=0)
        else:
            processed_signal = ppg_standardized
        
        # Normalize for CNN
        cnn_signal = (processed_signal - np.mean(processed_signal)) / (np.std(processed_signal) + 1e-8)
        
        # Extract CNN features
        if cnn_model is not None:
            with torch.no_grad():
                signal_tensor = torch.FloatTensor(cnn_signal).unsqueeze(0).unsqueeze(0).to(device)
                cnn_features = cnn_model.extract_features_only(signal_tensor)
                cnn_features_np = cnn_features.cpu().numpy().flatten()
        else:
            # If no CNN model, create dummy features (zeros)
            cnn_features_np = np.zeros(32)
            print("no CNN features!")
        
        # ===== TIME-DOMAIN FEATURE EXTRACTION =====
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
        
        # Exclude the pnn50 and pnn20 from the TD features
        if td_stats is not None:
            td_stats.pop('pNN50', None)
            td_stats.pop('pNN20', None)
            td_stats.pop('label', None)

        end_time = time.perf_counter()
        inference_time = end_time - start_time

        # Convert TD stats to list
        if isinstance(td_stats, dict):
            td_features = list(td_stats.values())
        else:
            td_features = td_stats
        
        # Combine CNN and TD features
        return list(cnn_features_np) + td_features, inference_time
        
    except Exception as e:
        print(f"Error extracting hybrid features: {e}")
        return None

#%% Test and measure complete hybrid inference
    
# Now measure complete hybrid inference time
n_inference_tests = 100
inference_times = []

print(f"Running {n_inference_tests} hybrid inference tests...")

for i in range(n_inference_tests):
    
    # Step 1: Extract hybrid features (CNN + TD) from raw PPG
    hybrid_features, feature_extraction_time = extract_hybrid_features_for_inference_sample(
        inference_sample_metadata, cnn_model, device
    )

    model_prediction_start_time = time.perf_counter()

    if hybrid_features is not None:
        
        # Step 3: Combine all features (CNN + TD)
        combined_features = hybrid_features
        feature_vector = np.array(combined_features).reshape(1, -1)
        
        # Step 4: Scale features using the same scaler from training
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Step 5: Make prediction
        prediction = final_model.predict(feature_vector_scaled)
        
    model_prediction_end_time = time.perf_counter()
    model_prediction_time = model_prediction_end_time - model_prediction_start_time
    inference_times.append((feature_extraction_time + model_prediction_time) * 1000)  # Convert to milliseconds

avg_inference_time = np.mean(inference_times)
std_inference_time = np.std(inference_times)

print(f"Complete hybrid inference measurement completed")

# %% Save results

results = {
    "training_time_s": training_time,
    "model_size_KB": model_size_kb,
    "total_deployment_size_KB": total_deployment_size_kb,
    "average_inference_time_ms": avg_inference_time,
    "std_inference_time_ms": std_inference_time,
}


output_csv_path = "AKTIVES_hybrid_inference.csv"

save_df = pd.DataFrame([results])
save_df.to_csv(output_csv_path, index=False)

# %%
