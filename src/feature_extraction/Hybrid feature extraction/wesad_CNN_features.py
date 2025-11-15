"""
WESAD Dataset Hybrid Processing Pipeline

Processes the WESAD dataset for stress classification using PPG signals.
Implements a hybrid feature extraction approach combining CNN-based features
and traditional time-domain HRV features.
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import glob

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter

#%%
class Simple1DCNNFeatures(nn.Module):
    """Simple CNN for feature extraction from PPG signals"""
    def __init__(self, input_length=7680, output_features=32):
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
        
    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        return outputs
    
    def extract_features_only(self, x):
        """Extract only features (for hybrid approach)"""
        return self.feature_extractor(x)

def prepare_training_data_for_cnn(subject_data_list, window_samples, step_samples):
    """Prepare training data from multiple subjects for CNN training"""
    all_windows = []
    all_labels = []
    
    for df in subject_data_list:
        print(f"   Processing subject data: {len(df)} samples")
        
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

def train_global_cnn(training_subjects, device, epochs=30):
    """Train CNN globally on fixed set of subjects"""
    DATA_PATH = '../data/WESAD_BVP_extracted/'
    WINDOW_SAMPLES = 120 * 64  # 120s at 64Hz
    STEP_SAMPLES = 30 * 64     # 30s step
    
    print(f"Training CNN on subjects: {training_subjects}")
    
    # Load training data
    training_data_list = []
    for train_subj in training_subjects:
        train_file = os.path.join(DATA_PATH, f'S{train_subj}.csv')
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file)
            training_data_list.append(train_df)
            print(f"   Loaded S{train_subj}: {len(train_df)} samples")
    
    if len(training_data_list) == 0:
        raise Exception("No training data found!")
    
    # Prepare training windows
    X_train, y_train = prepare_training_data_for_cnn(training_data_list, WINDOW_SAMPLES, STEP_SAMPLES)
    print(f"Training data: {len(X_train)} windows, labels: {np.bincount(y_train)}")
    
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop (simplified for 1 epoch)
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
        print(f"   Epoch {epoch+1}: Val Acc: {val_acc:.1f}%")
    
    print(f"CNN training completed. Final val acc: {val_acc:.1f}%")
    return model

def extract_features_for_subject(subject_id, trained_model, device):
    """Extract hybrid features for one subject using pre-trained CNN"""
    DATA_PATH = '../data/WESAD_BVP_extracted/'
    WINDOW_SAMPLES = 120 * 64
    STEP_SAMPLES = 30 * 64
    
    print(f"Extracting features for S{subject_id}...")
    
    # Load subject data
    subject_file = os.path.join(DATA_PATH, f'S{subject_id}.csv')
    if not os.path.exists(subject_file):
        print(f"File not found: {subject_file}")
        return None
    
    df = pd.read_csv(subject_file)
    
    # Extract features for both labels
    all_cnn_features = []
    all_td_features = []
    all_labels = []
    all_subject_ids = []
    
    trained_model.eval()
    
    for label_value in [0.0, 1.0]:
        label_data = df[df['label'] == label_value]['BVP'].values
        
        if len(label_data) < WINDOW_SAMPLES:
            print(f"   Warning: Not enough data for label {label_value}")
            continue
        
        # Create windows
        for start_idx in range(0, len(label_data) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = label_data[start_idx:start_idx + WINDOW_SAMPLES]
            
            try:
                # CNN feature extraction
                window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                window_tensor = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    cnn_features = trained_model.extract_features_only(window_tensor)
                    cnn_features_np = cnn_features.cpu().numpy().flatten()
                
                # Time-domain feature extraction
                bp_bvp = bandpass_filter(window, 0.2, 10, 64, order=2)
                smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
                
                td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                          fs=64, 
                                          label=int(label_value), 
                                          calc_sq=False)
                
                # Store if successful
                if td_stats and len(cnn_features_np) == 32:
                    all_cnn_features.append(cnn_features_np)
                    all_td_features.append(td_stats)
                    all_labels.append(int(label_value))
                    all_subject_ids.append(subject_id)
                    
            except Exception as e:
                print(f"   Error processing window: {e}")
                continue
    
    if not all_cnn_features:
        print(f"   No features extracted for S{subject_id}")
        return None
    
    print(f"   Extracted {len(all_cnn_features)} windows")
    
    # Create combined feature dataframe
    cnn_array = np.array(all_cnn_features)
    
    # Handle TD features
    if isinstance(all_td_features[0], dict):
        td_df_temp = pd.DataFrame(all_td_features)
        td_array = td_df_temp.values
        td_columns = list(td_df_temp.columns)
    else:
        td_array = np.array(all_td_features)
        td_columns = [f'td_feature_{i}' for i in range(td_array.shape[1])]
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    cnn_scaler = StandardScaler()
    td_scaler = StandardScaler()
    
    cnn_standardized = cnn_scaler.fit_transform(cnn_array)
    td_standardized = td_scaler.fit_transform(td_array)
    
    # Create dataframes
    cnn_columns = [f'cnn_feature_{i:04d}' for i in range(32)]
    cnn_df = pd.DataFrame(cnn_standardized, columns=cnn_columns)
    td_df = pd.DataFrame(td_standardized, columns=td_columns)
    
    # Combine all data
    result_df = pd.concat([
        pd.DataFrame({'subject_id': all_subject_ids, 'label': all_labels}),
        cnn_df,
        td_df
    ], axis=1)
    
    return result_df

def run_global_hybrid_extraction_with_timing():
    """Main function with complete timing measurement"""
    
    # Configuration
    cnn_training_subjects = [2, 3, 4, 5, 6]  # Fixed CNN training subjects
    all_subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # All subjects for feature extraction
    output_file = 'data/all_subjects_WESAD_hybrid_features_timing.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== TIMING STARTS HERE =====
    total_start_time = time.perf_counter()
    
    print("="*70)
    print("GLOBAL CNN HYBRID FEATURE EXTRACTION WITH TIMING")
    print("="*70)
    
    # Step 1: Train CNN globally (ONE TIME)
    print(f"\n1. Training CNN globally on subjects {cnn_training_subjects}")
    cnn_training_start = time.perf_counter()
    
    try:
        trained_model = train_global_cnn(cnn_training_subjects, device, epochs=30)
        cnn_training_time = time.perf_counter() - cnn_training_start
        print(f"CNN training completed in {cnn_training_time:.2f} seconds")
    except Exception as e:
        print(f"CNN training failed: {e}")
        return None
    
    # Step 2: Extract features for all subjects
    print(f"\n2. Extracting features for all subjects using trained CNN")
    feature_extraction_start = time.perf_counter()
    
    all_dataframes = []
    successful_subjects = []
    failed_subjects = []
    
    for subject_id in all_subjects:
        try:
            subject_df = extract_features_for_subject(subject_id, trained_model, device)
            if subject_df is not None:
                all_dataframes.append(subject_df)
                successful_subjects.append(subject_id)
                print(f"   S{subject_id}: {len(subject_df)} windows extracted")
            else:
                failed_subjects.append(subject_id)
                print(f"   S{subject_id}: FAILED")
        except Exception as e:
            failed_subjects.append(subject_id)
            print(f"   S{subject_id}: ERROR - {e}")
    
    feature_extraction_time = time.perf_counter() - feature_extraction_start
    print(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")
    
    # Step 3: Combine all dataframes
    if all_dataframes:
        print(f"\n3. Combining {len(all_dataframes)} subject dataframes...")
        combine_start = time.perf_counter()
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
     
        # Remove duplicate columns if any
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # Save combined file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        
        combine_time = time.perf_counter() - combine_start
        print(f"Combination completed in {combine_time:.2f} seconds")
    else:
        print("No successful extractions to combine!")
        return None
    
    # ===== TIMING ENDS HERE =====
    total_time = time.perf_counter() - total_start_time
    
    # Final Summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"CNN Training Time:        {cnn_training_time:.3f} seconds")
    print(f"Feature Extraction Time:  {feature_extraction_time:.3f} seconds")
    print(f"Data Combination Time:    {combine_time:.3f} seconds")
    print(f"TOTAL TIME:              {total_time:.3f} seconds")
    
    print("\nExtraction Summary:")
    print(f"Successful subjects: {len(successful_subjects)} - {successful_subjects}")
    if failed_subjects:
        print(f"Failed subjects: {len(failed_subjects)} - {failed_subjects}")
    
    print(f"\nFinal dataset:")
    print(f"Shape: {combined_df.shape}")
    print(f"Output file: {output_file}")
    
    if 'label' in combined_df.columns:
        label_dist = combined_df['label'].value_counts().sort_index()
        print(f"Label distribution: {dict(label_dist)}")
    
    if 'subject_id' in combined_df.columns:
        subject_dist = combined_df['subject_id'].value_counts().sort_index()
        print(f"Subject distribution: {dict(subject_dist)}")
    
    # Return timing information for comparison
    return {
        'total_time': total_time,
        'cnn_training_time': cnn_training_time,
        'feature_extraction_time': feature_extraction_time,
        'combine_time': combine_time,
        'successful_subjects': len(successful_subjects),
        'total_subjects': len(all_subjects),
        'final_dataset_shape': combined_df.shape
    }

# Run the full pipeline
if __name__ == "__main__":
    results = run_global_hybrid_extraction_with_timing()
    if results:
        print(f"\nPipeline completed successfully!")
        print(f"Total processing time: {results['total_time']:.2f} seconds")
    else:
        print(f"\nPipeline failed!")
# %%
