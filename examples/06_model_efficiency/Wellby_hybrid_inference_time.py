# Wellby hybrid efficiency calculation using comprehensive timing approach

import time
import os
import psutil
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn import svm
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('../..')
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import *

# Configuration
output_filename = 'wellby_hybrid_inference.csv'

class Simple1DCNNFeatures(nn.Module):
    def __init__(self, input_length=3000, output_features=32):
        super(Simple1DCNNFeatures, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
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
    def __init__(self, input_length=3000, num_classes=2):
        super().__init__()
        self.feature_extractor = Simple1DCNNFeatures(input_length)
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        return outputs
    
    def extract_features_only(self, x):
        return self.feature_extractor(x)

def load_wellby_data():
    print("Loading Wellby data...")
    
    ppg_data = pd.read_csv("../../data/Wellby/selected_PPG_data.csv")
    session_info = pd.read_csv("../../data/Wellby/Wellby_all_subjects_features.csv")
    
    ppg_session_ids = ppg_data.columns.tolist()
    info_session_ids = session_info['Session_ID'].tolist()
    
    common_sessions = list(set(ppg_session_ids) & set(info_session_ids))
    
    if len(common_sessions) == 0:
        return None
    
    aligned_data = []
    
    for session_id in common_sessions:
        ppg_signal = ppg_data[session_id].dropna().values
        session_row = session_info[session_info['Session_ID'] == session_id].iloc[0]
        
        aligned_data.append({
            'session_id': session_id,
            'participant': session_row['Participant'],
            'ppg_signal': ppg_signal,
            'stress_label': session_row['stress_binary'],
            'school': session_row['School'],
            'age': session_row['Age'],
            'gender': session_row['Gender'],
            'PSS': session_row['PSS'],
            'PSQI': session_row['PSQI'],
            'EPOCH': session_row['EPOCH'],
            'SQI': session_row['SQI']
        })
    
    return aligned_data

def prepare_wellby_signals(aligned_data, target_length=None, skip_start_samples=100):
    if target_length is None:
        signal_lengths = [len(item['ppg_signal']) - skip_start_samples for item in aligned_data 
                         if len(item['ppg_signal']) > skip_start_samples]
        if len(signal_lengths) == 0:
            return []
        target_length = min(signal_lengths)
    
    processed_data = []
    
    for item in aligned_data:
        ppg_signal = item['ppg_signal']
        
        if len(ppg_signal) <= skip_start_samples:
            continue
            
        trimmed_signal = ppg_signal[skip_start_samples:]
        
        if len(trimmed_signal) >= target_length:
            processed_signal = trimmed_signal[:target_length]
        else:
            padding = target_length - len(trimmed_signal)
            processed_signal = np.pad(trimmed_signal, (0, padding), mode='constant', constant_values=0)
        
        clean_signal = processed_signal[~np.isnan(processed_signal)]
        if len(clean_signal) == 0:
            continue
            
        standardized_signal = standardize(clean_signal)
        normalized_signal = (standardized_signal - np.mean(standardized_signal)) / (np.std(standardized_signal) + 1e-8)
        
        item_copy = item.copy()
        item_copy['cnn_signal'] = normalized_signal
        item_copy['raw_signal'] = trimmed_signal
        item_copy['signal_length'] = target_length
        
        processed_data.append(item_copy)
    
    return processed_data

def train_wellby_cnn_with_cv(processed_data, device, epochs=30):
    participants = [item['participant'] for item in processed_data]
    labels = [item['stress_label'] for item in processed_data]
    signals = [item['cnn_signal'] for item in processed_data]
    
    X = np.array(signals)
    y = np.array(labels)
    groups = np.array(participants)
    
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    trained_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        model = TrainableCNNFeatureExtractor(input_length=len(X_train[0])).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        torch.save(model.state_dict(), f'temp_best_wellby_cnn_fold_{fold_idx}.pth')
        
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
            
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'temp_best_wellby_cnn_fold_{fold_idx}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(torch.load(f'temp_best_wellby_cnn_fold_{fold_idx}.pth'))
        trained_models.append(model)
        
        torch.cuda.empty_cache()
    
    return trained_models

def extract_wellby_features(processed_data, trained_models, fs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_list = []
    successful_extractions = 0
    
    # Start timing feature extraction
    start_time_feat_extraction = time.perf_counter()
    
    for item in processed_data:
        try:
            # CNN features - ensemble average
            cnn_features_ensemble = []
            
            for model in trained_models:
                model.eval()
                with torch.no_grad():
                    signal_tensor = torch.FloatTensor(item['cnn_signal']).unsqueeze(0).unsqueeze(0).to(device)
                    cnn_features = model.extract_features_only(signal_tensor)
                    cnn_features_np = cnn_features.cpu().numpy().flatten()
                    cnn_features_ensemble.append(cnn_features_np)
            
            cnn_features_avg = np.mean(cnn_features_ensemble, axis=0)
            
            # TD features
            raw_signal = item['raw_signal']
            clean_signal = raw_signal[~np.isnan(raw_signal)]
            if len(clean_signal) == 0:
                continue
                
            standardized_signal = standardize(clean_signal)
            bp_signal = bandpass_filter(standardized_signal, 0.5, 10, fs, order=2)
            smoothed_signal = moving_average_filter(bp_signal, window_size=5)
            
            td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                      fs=fs, 
                                      label=item['stress_label'], 
                                      calc_sq=False)
            
            if td_stats is None or len(cnn_features_avg) != 32:
                continue
            
            # Combine features
            features_dict = {
                'session_id': item['session_id'],
                'participant': item['participant'],
                'label': item['stress_label'],
                'school': item['school'],
                'age': item['age'],
                'gender': item['gender'],
                'PSS': item['PSS'],
                'PSQI': item['PSQI'],
                'EPOCH': item['EPOCH'],
                'SQI': item['SQI']
            }
            
            # Add CNN features
            for i, feat in enumerate(cnn_features_avg):
                features_dict[f'cnn_feature_{i:04d}'] = feat
            
            # Add TD features
            if isinstance(td_stats, dict):
                features_dict.update(td_stats)
            else:
                td_names = ['mean_hr', 'std_hr', 'rmssd', 'sdnn', 'sdsd', 
                           'mean_nn', 'mean_sd', 'median_nn', 'pnn20', 'pnn50']
                for i, feat in enumerate(td_stats[:len(td_names)]):
                    features_dict[td_names[i]] = feat
            
            features_list.append(features_dict)
            successful_extractions += 1
            
        except Exception as e:
            continue
    
    feature_extraction_time = time.perf_counter() - start_time_feat_extraction
    
    return features_list, feature_extraction_time, successful_extractions

def main():
    print("="*70)
    print("WELLBY HYBRID EFFICIENCY ANALYSIS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===== TIMING STARTS =====
    total_start_time = time.perf_counter()
    
    # Step 1: Load data
    data_loading_start = time.perf_counter()
    aligned_data = load_wellby_data()
    if aligned_data is None:
        print("Error: Could not load Wellby data!")
        return None
    data_loading_time = time.perf_counter() - data_loading_start
    
    # Step 2: Prepare signals
    processed_data = prepare_wellby_signals(aligned_data)
    if len(processed_data) == 0:
        print("Error: No signals could be processed!")
        return None
    
    # Reserve one sample for inference testing (exclude from training)
    np.random.seed(42)
    test_sample_idx = np.random.choice(len(processed_data), 1)[0]
    test_sample = processed_data[test_sample_idx]
    
    # Remove test sample from training data
    training_data = [item for i, item in enumerate(processed_data) if i != test_sample_idx]
    
    print(f"Reserved sample {test_sample_idx} for inference testing")
    print(f"Training on {len(training_data)} samples")
    
    # Step 3: Train CNN
    print("Training CNN...")
    cnn_training_start = time.perf_counter()
    trained_models = train_wellby_cnn_with_cv(training_data, device)
    cnn_training_time = time.perf_counter() - cnn_training_start
    
    # Step 4: Extract features (using training data only)
    print("Extracting features...")
    features_list, feature_extraction_time, successful_extractions = extract_wellby_features(training_data, trained_models)
    
    if len(features_list) == 0:
        print("No features extracted!")
        return None
    
    # Step 5: Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    cnn_columns = [col for col in features_df.columns if col.startswith('cnn_feature_')]
    td_columns = [col for col in features_df.columns if col not in cnn_columns and 
                  col not in ['session_id', 'participant', 'label', 'school', 'age', 'gender', 'PSS', 'PSQI', 'EPOCH', 'SQI']]
    
    # Standardize features
    scaler_cnn = StandardScaler()
    scaler_td = StandardScaler()
    
    features_df[cnn_columns] = scaler_cnn.fit_transform(features_df[cnn_columns])
    if len(td_columns) > 0:
        features_df[td_columns] = scaler_td.fit_transform(features_df[td_columns])
    
    # Step 6: Train SVM
    print("Training SVM...")
    svm_training_start = time.perf_counter()
    
    all_feature_cols = cnn_columns + td_columns
    X = features_df[all_feature_cols].values
    y = features_df['label'].values
    participants = features_df['participant'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'coef0': [0.0, 0.1, 0.5, 1.0]
    }
    
    svm_model = svm.SVC(probability=True, class_weight='balanced', random_state=42)
    
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    grid_search = GridSearchCV(
        estimator=svm_model, 
        param_grid=param_grid, 
        cv=inner_cv, 
        scoring='roc_auc', 
        n_jobs=1
    )
    grid_search.fit(X_scaled, y, groups=participants)
    
    final_model = svm.SVC(**grid_search.best_params_, probability=True, class_weight='balanced', random_state=42)
    final_model.fit(X_scaled, y)
    svm_training_time = time.perf_counter() - svm_training_start
    
    # Step 7: Measure complete inference time (raw PPG -> features -> prediction)
    print("Measuring complete inference time...")
    
    # Reserve one sample for inference testing (exclude from training)
    np.random.seed(42)
    test_sample_idx = np.random.choice(len(processed_data), 1)[0]
    test_sample = processed_data[test_sample_idx]
    
    # Get raw PPG signal for inference test
    test_raw_ppg = test_sample['ppg_signal']
    test_participant = test_sample['participant']
    
    inference_times = []
    successful_inferences = 0
    
    for i in range(100):
        start_time = time.perf_counter()
        
        try:
            # Step 1: Process raw PPG for CNN
            if len(test_raw_ppg) <= 100:  # skip_start_samples
                continue
            trimmed_signal = test_raw_ppg[100:]  # Skip start samples
            
            # Handle length (same logic as prepare_wellby_signals)
            target_length = test_sample['signal_length']
            if len(trimmed_signal) >= target_length:
                processed_signal = trimmed_signal[:target_length]
            else:
                padding = target_length - len(trimmed_signal)
                processed_signal = np.pad(trimmed_signal, (0, padding), mode='constant', constant_values=0)
            
            clean_signal = processed_signal[~np.isnan(processed_signal)]
            if len(clean_signal) == 0:
                continue
            
            standardized_signal = standardize(clean_signal)
            cnn_signal = (standardized_signal - np.mean(standardized_signal)) / (np.std(standardized_signal) + 1e-8)
            
            # Step 2: Extract CNN features (ensemble average)
            cnn_features_ensemble = []
            for model in trained_models:
                model.eval()
                with torch.no_grad():
                    signal_tensor = torch.FloatTensor(cnn_signal).unsqueeze(0).unsqueeze(0).to(device)
                    cnn_features = model.extract_features_only(signal_tensor)
                    cnn_features_np = cnn_features.cpu().numpy().flatten()
                    cnn_features_ensemble.append(cnn_features_np)
            
            cnn_features_avg = np.mean(cnn_features_ensemble, axis=0)
            
            # Step 3: Extract TD features
            clean_signal_td = trimmed_signal[~np.isnan(trimmed_signal)]
            standardized_signal_td = standardize(clean_signal_td)
            bp_signal = bandpass_filter(standardized_signal_td, 0.5, 10, fs=50, order=2)
            smoothed_signal = moving_average_filter(bp_signal, window_size=5)

            segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95, window_size=3)
            sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
            sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
            
            td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                      fs=50, 
                                      label=0,  # Dummy label
                                      calc_sq=False)
            
            if td_stats is None:
                continue
                
            # Step 4: Combine features
            if isinstance(td_stats, dict):
                td_stats.pop('label', None)
                td_features = list(td_stats.values())
            else:
                td_features = td_stats
            
            combined_features = np.concatenate([cnn_features_avg, td_features]).reshape(1, -1)
            
            # Step 5: Scale features
            combined_features_scaled = scaler.transform(combined_features)
            
            # Step 6: Make prediction
            prediction = final_model.predict(combined_features_scaled)
            
            successful_inferences += 1
            
        except Exception as e:
            # Use fallback for failed inference
            prediction = [0]
        
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"Complete inference measurement completed!")
    print(f"Successful inferences: {successful_inferences}/100")
    print(f"Average complete inference time: {avg_inference_time:.3f}ms")
    print(f"Standard deviation: {std_inference_time:.3f}ms")
    
    # Model sizes
    model_size_bytes = len(pickle.dumps(final_model))
    scaler_size_bytes = len(pickle.dumps(scaler))
    
    # Save and measure CNN model size
    torch.save(trained_models[0].state_dict(), 'temp_cnn_wellby.pth')
    cnn_size_bytes = os.path.getsize('temp_cnn_wellby.pth')
    os.remove('temp_cnn_wellby.pth')
    
    total_deployment_size_kb = (model_size_bytes + scaler_size_bytes + cnn_size_bytes) / 1024
    
    # ===== TIMING ENDS =====
    total_time = time.perf_counter() - total_start_time
    
    # Results
    results = {
        "total_time_s": total_time,
        "data_loading_time_s": data_loading_time,
        "cnn_training_time_s": cnn_training_time,
        "feature_extraction_time_s": feature_extraction_time,
        "svm_training_time_s": svm_training_time,
        "average_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "successful_inferences": successful_inferences,
        "svm_model_size_kb": model_size_bytes / 1024,
        "scaler_size_kb": scaler_size_bytes / 1024,
        "cnn_model_size_kb": cnn_size_bytes / 1024,
        "total_deployment_size_kb": total_deployment_size_kb,
    }
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_filename, index=False)
    
    # Cleanup
    for i in range(3):
        temp_file = f'temp_best_wellby_cnn_fold_{i}.pth'
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return results

if __name__ == "__main__":
    results = main()