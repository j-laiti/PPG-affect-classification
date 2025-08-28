#%% Efficiency Measurement Script for WESAD Hybrid Features (CNN + TD)
import time
import sys
import os
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy import stats
import csv

# Add path for preprocessing functions
sys.path.append('../../src/')
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize

# ===== CNN CLASSES (from your hybrid code) =====
class Simple1DCNNFeatures(nn.Module):
    """Simple CNN for feature extraction from PPG signals"""
    def __init__(self, input_length=7680, output_features=32):
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
    def __init__(self, input_length=7680, num_classes=2):
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

# ===== HELPER FUNCTIONS =====
def calculate_confidence_intervals(scores, confidence_level=0.95):
    """Calculate confidence intervals for cross-validation results"""
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(n)
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se
    
    return {
        'mean': mean, 'std': std, 'se': se,
        'ci_lower': ci_lower, 'ci_upper': ci_upper
    }

def read_hybrid_csv(path, testset_num):
    """Read hybrid CSV and split into train/test"""
    print(f"Loading data from: {path}")
    print(f"Test subject: S{testset_num}")
    
    df = pd.read_csv(path)
    print(f"Total dataset shape: {df.shape}")
    
    # Get feature columns (exclude subject_id and label)
    non_feature_cols = ['subject_id', 'label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"CNN features: {len([col for col in feature_cols if 'cnn_feature_' in col])}")
    print(f"TD features: {len([col for col in feature_cols if 'cnn_feature_' not in col])}")
    
    # Split by subject
    train_df = df[df['subject_id'] != testset_num].copy()
    test_df = df[df['subject_id'] == testset_num].copy()
    
    print(f"Train subjects: {sorted(train_df['subject_id'].unique())}")
    print(f"Test subject: {sorted(test_df['subject_id'].unique())}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    if len(test_df) == 0:
        print(f"WARNING: No data found for test subject S{testset_num}")
        return None, None, None, None, None
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def SVM_model_with_timing(X_train, y_train, X_test, y_test):
    """SVM model with timing measurements (no GridSearch - consistent with WESAD TD approach)"""
    # Measure training time
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Simple SVM (matching your existing WESAD approach)
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
    
    # Measure scaler size
    scaler_size_bytes = len(pickle.dumps(StandardScaler().fit(X_train)))
    scaler_size_kb = scaler_size_bytes / 1024
    total_deployment_size_kb = model_size_kb + scaler_size_kb
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    
    return AUC, F1, accuracy, training_time, model_size_kb, total_deployment_size_kb, model

def prepare_training_data_for_cnn(subject_data_list, window_samples, step_samples):
    """Prepare training data from multiple subjects for CNN training"""
    all_windows = []
    all_labels = []
    
    for df in subject_data_list:
        print(f"     Processing subject data: {len(df)} samples")
        
        # Process stress data (label = 1.0)
        stress_data = df[df['label'] == 1.0]['BVP'].values
        if len(stress_data) >= window_samples:
            for start_idx in range(0, len(stress_data) - window_samples + 1, step_samples):
                window = stress_data[start_idx:start_idx + window_samples]
                if len(window) == window_samples:
                    window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    all_windows.append(window_norm)
                    all_labels.append(1)
        
        # Process non-stress data (label = 0.0)
        non_stress_data = df[df['label'] == 0.0]['BVP'].values
        if len(non_stress_data) >= window_samples:
            for start_idx in range(0, len(non_stress_data) - window_samples + 1, step_samples):
                window = non_stress_data[start_idx:start_idx + window_samples]
                if len(window) == window_samples:
                    window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    all_windows.append(window_norm)
                    all_labels.append(0)
    
    return np.array(all_windows), np.array(all_labels)

def train_cnn_for_efficiency_test(X_train, y_train, device):
    """Quick CNN training for efficiency testing (reduced epochs)"""
    print(f"     Training CNN: {len(X_train)} windows, labels: {np.bincount(y_train)}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_tensor = torch.LongTensor(y_train)
    
    # Simple train/val split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = TrainableCNNFeatureExtractor().to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Quick training (reduced epochs for efficiency testing)
    epochs = 15  # Reduced from 30
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = 100 * val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'temp_best_cnn_s{device.type}.pth')
            
            print(f"       Epoch {epoch+1}: Val Acc: {val_acc:.1f}%")
        
        torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load(f'temp_best_cnn_s{device.type}.pth'))
    print(f"     CNN training completed. Best val acc: {best_val_acc:.1f}%")
    
    return model

def extract_hybrid_features_for_inference_sample(test_data_path, trained_model, device):
    """Extract hybrid features for inference timing from test CSV file"""
    # Configuration
    WINDOW_SIZE = 120
    SAMPLING_RATE = 64
    WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE
    
    try:
        # Load raw PPG data from CSV file (same as TD approach)
        raw_ppg_df = pd.read_csv(test_data_path)
        raw_ppg = raw_ppg_df.iloc[:, 0].values
        
        # Remove NaN values
        clean_ppg_values = raw_ppg[~np.isnan(raw_ppg)]
        
        if len(clean_ppg_values) < WINDOW_SAMPLES:
            # If signal is shorter than expected window, use what we have
            window = clean_ppg_values
        else:
            # Extract window of expected size
            window = clean_ppg_values[:WINDOW_SAMPLES]
        
        # === PREPROCESSING PIPELINE (same as TD approach) ===
        #  Standardize
        ppg_standardized = standardize(window)
        
        # Apply filtering
        bp_bvp = bandpass_filter(ppg_standardized, 0.5, 10, 64, order=2)
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
        
        # Ensure we have enough data for both CNN and TD feature extraction
        if len(smoothed_signal) < 640:  # Need at least 10 seconds for TD features
            print(f"Warning: Signal too short after preprocessing: {len(smoothed_signal)} samples")
            return None
            
        # === CNN FEATURE EXTRACTION ===
        # Prepare signal for CNN (need exactly WINDOW_SAMPLES)
        if len(smoothed_signal) >= WINDOW_SAMPLES:
            cnn_window = smoothed_signal[:WINDOW_SAMPLES]
        else:
            # Pad if necessary
            cnn_window = np.pad(smoothed_signal, (0, WINDOW_SAMPLES - len(smoothed_signal)), mode='edge')

        window_norm = (cnn_window - np.mean(cnn_window)) / (np.std(cnn_window) + 1e-8)
        window_tensor = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cnn_features = trained_model.extract_features_only(window_tensor)
            cnn_features_np = cnn_features.cpu().numpy().flatten()
        
        # === TIME-DOMAIN FEATURE EXTRACTION ===
        td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                  fs=64, 
                                  label=1, 
                                  calc_sq=True)
        
        if td_stats is None or len(td_stats) <= 1:
            print("Warning: TD feature extraction failed")
            return None
            
        # Remove label and handle SQI
        if 'sqi' in td_stats:
            td_stats['SQI'] = td_stats.pop('sqi')
        td_stats.pop('label', None)
        
        # Convert TD stats to array (matching the feature order from training)
        td_feature_names = ['HR_mean','HR_std','meanNN','SDNN','medianNN','meanSD','SDSD','RMSSD','pNN20','pNN50']
        if 'SQI' in td_stats:
            td_feature_names.append('SQI')
            
        td_features = [td_stats.get(feat, 0.0) for feat in td_feature_names]
        
        # === COMBINE FEATURES ===
        # CNN features first, then TD features (matching training order)
        combined_features = list(cnn_features_np) + td_features
        
        return combined_features
        
    except Exception as e:
        print(f"Error extracting hybrid features: {e}")
        import traceback
        traceback.print_exc()
        return None

def measure_hybrid_inference_time(test_data_path, trained_model, svm_model, scaler, device, n_tests=100):
    """Measure complete hybrid inference time including preprocessing, CNN, TD features, and SVM prediction"""
    print(f"   Measuring hybrid inference time with test file: {test_data_path}")
    
    inference_times = []
    successful_inferences = 0
    
    for i in range(n_tests):
        start_time = time.perf_counter()
        
        try:
            # Step 1: Extract hybrid features (includes full preprocessing pipeline + CNN + TD)
            hybrid_features = extract_hybrid_features_for_inference_sample(test_data_path, trained_model, device)
            
            if hybrid_features is not None:
                # Step 2: Scale features
                feature_vector = np.array(hybrid_features).reshape(1, -1)
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Step 3: Make SVM prediction
                prediction = svm_model.predict(feature_vector_scaled)
                
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
    
    print(f"✓ Hybrid inference measurement completed! ({successful_inferences}/{n_tests} successful)")
    print(f"  Average inference time: {avg_inference_time:.3f}ms")
    print(f"  Standard deviation: {std_inference_time:.3f}ms")
    
    return avg_inference_time, std_inference_time

# Update the main evaluation function to use the test CSV file
def run_hybrid_efficiency_evaluation():
    print("\n" + "="*50)
    print("WESAD HYBRID FEATURES - EFFICIENCY MEASUREMENT")
    print("="*50)
    
    # Configuration
    COMBINED_DATA_PATH = '../../data/all_subjects_WESAD_hybrid_features.csv'
    BVP_DATA_PATH = '../../data/WESAD_BVP_extracted/'
    TEST_DATA_PATH = '../../data/WESAD_inference_test_signal.csv'  # Add test signal path
    RESULTS_PATH_ALL = '../../results/WESAD/Efficiency/hybrid_results.csv'
    RESULTS_PATH_DETAILED = '../../results/WESAD/Efficiency/hybrid_efficiency_detailed.csv'
    
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train CNN ONCE for all inference timing (not per LOGO fold)
    print("Training CNN once for inference timing...")
    trained_model = None
    try:
        # Load training data from first few subjects for CNN
        training_subjects = [2, 3, 4, 5, 6]  # Fixed set for CNN training
        training_data_list = []
        
        for train_subj in training_subjects:
            train_file = os.path.join(BVP_DATA_PATH, f'S{train_subj}.csv')
            if os.path.exists(train_file):
                train_df = pd.read_csv(train_file)
                training_data_list.append(train_df)
                print(f"  Loaded S{train_subj} for CNN training")
        
        if len(training_data_list) > 0:
            WINDOW_SAMPLES = 120 * 64  # 120s at 64Hz
            STEP_SAMPLES = 30 * 64     # 30s step
            
            X_cnn_train, y_cnn_train = prepare_training_data_for_cnn(training_data_list, WINDOW_SAMPLES, STEP_SAMPLES)
            
            if len(X_cnn_train) >= 50:
                trained_model = train_cnn_for_efficiency_test(X_cnn_train, y_cnn_train, device)
                print("✓ CNN trained successfully for inference timing")
            else:
                print("⚠ Insufficient CNN training data")
    except Exception as e:
        print(f"⚠ CNN training failed: {e}")
    
    # Storage for results
    SVM_AUC, SVM_F1, SVM_ACC = [], [], []
    training_times, model_sizes, deployment_sizes = [], [], []
    avg_inference_times, std_inference_times = [], []
    
    valid_subjects = []
    
    print(f"\nRunning LOGO validation for {len(subjects)} subjects...")
    
    for i, sub in enumerate(subjects):
        print(f"\n{'='*40}")
        print(f"Subject S{sub} ({i+1}/{len(subjects)})")
        print(f"{'='*40}")
        
        try:
            # Step 1: Load hybrid features for SVM training
            X_train, y_train, X_test, y_test, feature_cols = read_hybrid_csv(COMBINED_DATA_PATH, sub)
            
            if X_train is None:
                print(f"Skipping subject S{sub} - no data")
                continue
            
            # Handle missing values
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalization
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            # Step 2: Train SVM with timing (ONLY SVM training - hybrid features already extracted)
            print("Training SVM on pre-computed hybrid features...")
            auc_svm, f1_svm, acc_svm, train_time, model_size_kb, deploy_size_kb, svm_model = SVM_model_with_timing(
                X_train, y_train, X_test, y_test
            )
            
            # Step 3: Measure hybrid inference time using test CSV file
            if trained_model is not None and os.path.exists(TEST_DATA_PATH):
                print("Measuring hybrid inference time with test signal...")
                avg_inf_time, std_inf_time = measure_hybrid_inference_time(
                    TEST_DATA_PATH, trained_model, svm_model, sc, device
                )
            else:
                if trained_model is None:
                    print("No CNN model available - skipping inference timing")
                if not os.path.exists(TEST_DATA_PATH):
                    print(f"Test signal file not found: {TEST_DATA_PATH}")
                avg_inf_time, std_inf_time = np.nan, np.nan
            
            # Store results
            SVM_AUC.append(auc_svm * 100)
            SVM_F1.append(f1_svm * 100)
            SVM_ACC.append(acc_svm * 100)
            training_times.append(train_time)  # Only SVM training time
            model_sizes.append(model_size_kb)
            deployment_sizes.append(deploy_size_kb)
            avg_inference_times.append(avg_inf_time)
            std_inference_times.append(std_inf_time)
            
            valid_subjects.append(sub)
            
            print(f"Results for S{sub}:")
            print(f"  AUC: {auc_svm*100:.2f}%, F1: {f1_svm*100:.2f}%, Acc: {acc_svm*100:.2f}%")
            print(f"  SVM training time: {train_time:.3f}s")
            print(f"  Model: {model_size_kb:.2f}KB, Inference: {avg_inf_time:.3f}ms")
            
        except Exception as e:
            print(f"Error processing subject S{sub}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully processed {len(valid_subjects)} subjects: {valid_subjects}")
    
    # Calculate summary statistics (handle NaN values)
    valid_inference_times = [t for t in avg_inference_times if not np.isnan(t)]
    
    print(f"\n" + "="*80)
    print("WESAD HYBRID FEATURES - SUMMARY RESULTS")
    print("="*80)
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"AUC-ROC: {np.mean(SVM_AUC):.2f} ± {np.std(SVM_AUC):.2f}")
    print(f"F1 Score: {np.mean(SVM_F1):.2f} ± {np.std(SVM_F1):.2f}")
    print(f"Accuracy: {np.mean(SVM_ACC):.2f} ± {np.std(SVM_ACC):.2f}")
    
    print("\nEfficiency Metrics:")
    print("-" * 40)
    print(f"Avg SVM Training Time: {np.mean(training_times):.3f} ± {np.std(training_times):.3f} seconds")
    print(f"Avg Model Size: {np.mean(model_sizes):.2f} ± {np.std(model_sizes):.2f} KB")
    print(f"Avg Deployment Size: {np.mean(deployment_sizes):.2f} ± {np.std(deployment_sizes):.2f} KB")
    
    if valid_inference_times:
        print(f"Avg Inference Time: {np.mean(valid_inference_times):.3f} ± {np.std(valid_inference_times):.3f} ms")
    else:
        print("Avg Inference Time: No valid measurements")
    
    # Save results
    print(f"\nSaving results...")
    
    # Create results directory if it doesn't exist
    os.makedirs('../../results/WESAD/Efficiency', exist_ok=True)
    
    # Save efficiency details
    efficiency_results = pd.DataFrame({
        'subject': valid_subjects,
        'auc_roc': SVM_AUC,
        'f1_score': SVM_F1,
        'accuracy': SVM_ACC,
        'training_time_s': training_times,
        'model_size_KB': model_sizes,
        'deployment_size_KB': deployment_sizes,
        'avg_inference_time_ms': avg_inference_times,
        'std_inference_time_ms': std_inference_times
    })
    
    efficiency_results.to_csv(RESULTS_PATH_DETAILED, index=False)
    
    # Save summary for comparison with other approaches
    summary_results = {
        'approach': 'hybrid_cnn_td',
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
    summary_df.to_csv(RESULTS_PATH_ALL, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {RESULTS_PATH_ALL}")
    print(f"  - {RESULTS_PATH_DETAILED}")
    
    return efficiency_results