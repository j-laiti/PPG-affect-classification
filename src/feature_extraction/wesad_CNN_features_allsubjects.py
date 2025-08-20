#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter

class TrainableCNNFeatureExtractor(nn.Module):
    """
    Your existing CNN + classification head for training
    """
    def __init__(self, input_length=7680, num_classes=2):
        super().__init__()
        
        # 1D CNN architecture
        self.feature_extractor = Simple1DCNNFeatures(input_length)
        
        # Add classification head for training
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
    
def prepare_training_data_for_cnn(subject_data_list, window_samples, step_samples, fs):
    """
    Prepare training data from multiple subjects for CNN training
    
    Args:
        subject_data_list: List of dataframes from different subjects
        window_samples: Window size in samples
        step_samples: Step size in samples
        fs: Sampling frequency
    
    Returns:
        X_train: Windows for training
        y_train: Labels for training
    """
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
                    # Normalize window (same as your existing preprocessing)
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

def train_cnn_for_stress_detection(X_train, y_train, device, epochs=30, validation_split=0.2):
    """
    Train CNN for stress detection
    
    Args:
        X_train: Training windows (N, window_length)
        y_train: Training labels (N,)
        device: Training device
        epochs: Number of training epochs
        validation_split: Fraction for validation
    
    Returns:
        trained_model: Trained model for feature extraction
    """
    print(f"\n   Training CNN classifier...")
    print(f"   Training data: {len(X_train)} windows")
    print(f"   Label distribution: {np.bincount(y_train)}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_tensor = torch.LongTensor(y_train)
    
    # Train/validation split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tensor, y_tensor, test_size=validation_split, random_state=42, stratify=y_tensor
    )
    
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
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
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Print progress (reduced frequency)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'     Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'temp_best_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'     Early stopping at epoch {epoch+1}')
                break
        
        torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load('temp_best_cnn.pth'))
    print(f'   CNN training completed. Best validation accuracy: {best_val_acc:.1f}%')
    
    return model

class Simple1DCNNFeatures(nn.Module):
    """
    Simple CNN for feature extraction from PPG signals
    Based on Zhao et al. temporal convolutional approach
    """
    def __init__(self, input_length=7680, output_features=32):  # 120s at 64Hz
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection (kernel=15, ~0.23s)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Layer 2: Pattern refinement (kernel=11, ~0.17s)  
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Layer 3: High-level features (kernel=7, ~0.11s)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, output_features//4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # 32 features feature output
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x) 
        return x

def extract_features_for_label(ppg_signal, label_value, model, device, fs_dict, window_samples, step_samples):
    """
    Extract CNN and TD features for a specific label (0.0 or 1.0)
    """
    print(f"\n   Processing label {label_value} data...")
    print(f"   Signal length: {len(ppg_signal)} samples ({len(ppg_signal)/fs_dict['BVP']:.1f} seconds)")
    
    if len(ppg_signal) < window_samples:
        print(f"   Warning: Not enough data for even one window!")
        return [], [], []
    
    # Create windows
    windows = []
    for start_idx in range(0, len(ppg_signal) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window = ppg_signal[start_idx:end_idx]
        windows.append(window)
    
    print(f"   Created {len(windows)} windows")
    
    # Extract features for each window
    all_cnn_features = []
    all_td_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, window in enumerate(windows):
            try:
                # ===== CNN FEATURE EXTRACTION =====
                # Normalize window first
                window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
                window_tensor = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, window_length)
                window_tensor = window_tensor.to(device)
                
                
                # Extract CNN features using the trained model
                cnn_features = model.extract_features_only(window_tensor)  # Use the feature extraction method
                cnn_features_np = cnn_features.cpu().numpy().flatten()
                
                # ===== TIME-DOMAIN FEATURE EXTRACTION =====
                bp_bvp = bandpass_filter(window, 0.2, 10, fs_dict['BVP'], order=2)
                smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
                
                td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                          fs=fs_dict['BVP'], 
                                          label=int(label_value), 
                                          calc_sq=False)
                
                # Check if both extractions were successful
                if td_stats and len(td_stats) > 0 and len(cnn_features_np) == 32:  # Verify CNN features
                    all_cnn_features.append(cnn_features_np)
                    all_td_features.append(td_stats)
                    all_labels.append(label_value)
                    
                    if (i + 1) % 20 == 0 or (i + 1) == len(windows):
                        print(f"     Processed {i+1}/{len(windows)} windows ({len(all_cnn_features)} successful)")
                else:
                    print(f"     Window {i}: FAILED - CNN shape: {cnn_features_np.shape}, TD valid: {td_stats is not None}")
                        
            except Exception as e:
                print(f"     Error processing window {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"   Successfully extracted features from {len(all_cnn_features)}/{len(windows)} windows")
    return all_cnn_features, all_td_features, all_labels

def extract_combined_features_with_training(subject_id, training_subjects=None):
    """
    Modified version that includes CNN training
    
    Args:
        subject_id: Subject to extract features for
        training_subjects: List of subjects to train CNN on (if None, use global approach)
    """
    
    # Configuration (same as your original)
    DATA_PATH = '../../data/WESAD_BVP_extracted/'
    WINDOW_SIZE = 120
    STEP_SIZE = 30
    SAMPLING_RATE = 64
    WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE
    STEP_SAMPLES = STEP_SIZE * SAMPLING_RATE
    
    print("="*70)
    print(f"TRAINED CNN + TD FEATURES: S{subject_id}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: Prepare training data for CNN
    if training_subjects is None:
        # Use a fixed set of subjects for CNN training (Global approach)
        training_subjects = [2, 3, 4, 5, 6]  # First 5 subjects
    
    print(f"\n1. Preparing CNN training data from subjects: {training_subjects}")
    
    training_data_list = []
    for train_subj in training_subjects:
        if train_subj != subject_id:  # Don't include test subject in training
            train_file = os.path.join(DATA_PATH, f'S{train_subj}.csv')
            if os.path.exists(train_file):
                train_df = pd.read_csv(train_file)
                training_data_list.append(train_df)
                print(f"   Loaded S{train_subj}: {len(train_df)} samples")
    
    if len(training_data_list) == 0:
        print("   Error: No training data available!")
        return None
    
    # Step 2: Create training windows
    X_train, y_train = prepare_training_data_for_cnn(
        training_data_list, WINDOW_SAMPLES, STEP_SAMPLES, SAMPLING_RATE
    )
    
    if len(X_train) < 50:  # Need minimum amount of training data
        print(f"   Error: Insufficient training data ({len(X_train)} windows)")
        return None
    
    # Step 3: Train CNN
    trained_model = train_cnn_for_stress_detection(X_train, y_train, device)
    
    # Step 4: Load target subject data (same as your original)
    print(f"\n2. Loading target subject S{subject_id} data...")
    subject_file = os.path.join(DATA_PATH, f'S{subject_id}.csv')
    
    if not os.path.exists(subject_file):
        print(f"Error: File not found at {subject_file}")
        return None
    
    df = pd.read_csv(subject_file)
    
    # Step 5: Extract features using trained CNN (modify your existing extraction logic)
    print(f"\n3. Extracting features using trained CNN...")
    
    trained_model.eval()  # Set to evaluation mode
    
    # Process stress data
    stress_data = df[df['label'] == 1.0]
    non_stress_data = df[df['label'] == 0.0]
    
    fs_dict = {'BVP': SAMPLING_RATE}
    
    # Use your existing function but with trained model
    stress_cnn, stress_td, stress_labels = extract_features_for_label(
        stress_data['BVP'].values, 1.0, trained_model, device, fs_dict, WINDOW_SAMPLES, STEP_SAMPLES
    )
    
    non_stress_cnn, non_stress_td, non_stress_labels = extract_features_for_label(
        non_stress_data['BVP'].values, 0.0, trained_model, device, fs_dict, WINDOW_SAMPLES, STEP_SAMPLES
    )
    
    # Step 6: Combine and standardise all features
    print("\n6. Combining all features...")
    
    if len(stress_cnn) == 0 and len(non_stress_cnn) == 0:
        print("   Error: No successful feature extractions!")
        return None
    
    # Combine all CNN features
    all_cnn_features = stress_cnn + non_stress_cnn
    all_td_features = stress_td + non_stress_td  
    all_labels = stress_labels + non_stress_labels
    
    print(f"   Total windows: {len(all_cnn_features)}")
    print(f"   Stress windows: {len(stress_cnn)}")
    print(f"   Non-stress windows: {len(non_stress_cnn)}")

    cnn_feature_columns = [f'cnn_feature_{i:04d}' for i in range(32)]

    if len(all_cnn_features) > 0 and len(all_td_features) > 0:
        # Convert to arrays
        cnn_array = np.array(all_cnn_features)
        
        # Handle TD features - check if they're dictionaries or lists
        if isinstance(all_td_features[0], dict):
            # Convert dict to DataFrame first, then to array
            td_df_temp = pd.DataFrame(all_td_features)
            td_array = td_df_temp.values
            td_feature_columns = list(td_df_temp.columns)  # Use actual column names from dict
        else:
            # If it's a list, use predefined column names
            td_array = np.array(all_td_features)
            td_feature_columns = [
                'mean_hr', 'std_hr', 'rmssd', 'sdnn', 'sdsd', 
                'mean_nn', 'mean_sd', 'median_nn', 'pnn20', 'pnn50'
            ][:td_array.shape[1]]  # Trim to actual number of features
    
    # Standardize each feature type separately
    from sklearn.preprocessing import StandardScaler
    
    cnn_scaler = StandardScaler()
    td_scaler = StandardScaler()
    
    cnn_standardized = cnn_scaler.fit_transform(cnn_array)
    td_standardized = td_scaler.fit_transform(td_array)
    
    print(f"   CNN features - Before: mean={cnn_array.mean():.3f}, std={cnn_array.std():.3f}")
    print(f"   CNN features - After:  mean={cnn_standardized.mean():.3f}, std={cnn_standardized.std():.3f}")
    print(f"   TD features - Before:  mean={td_array.mean():.3f}, std={td_array.std():.3f}")
    print(f"   TD features - After:   mean={td_standardized.mean():.3f}, std={td_standardized.std():.3f}")
    
    # Create DataFrames with standardized data
    cnn_df = pd.DataFrame(cnn_standardized, columns=cnn_feature_columns)
    td_df = pd.DataFrame(td_standardized, columns=td_feature_columns)
    
    print(f"   TD feature columns: {td_feature_columns}")  # Debug print
    
    # Step 7: Create combined dataframe
    print("\n7. Creating combined dataframe...")
    
    # Create subject ID column (NEW: Add subject ID for each row)
    subject_ids = [subject_id] * len(all_cnn_features)
    
    # Create labels and subject ID dataframes
    labels_df = pd.DataFrame({'label': all_labels})
    subject_df = pd.DataFrame({'subject_id': subject_ids})  # NEW: Subject ID column
    
    # Combine: Subject ID + Labels + CNN Features + TD Features (NEW: Include subject_id first)
    results_df = pd.concat([subject_df, labels_df, cnn_df, td_df], axis=1)
    
    # Check for duplicate columns
    if results_df.columns.duplicated().any():
        print(f"   Warning: Found duplicate column names!")
        print(f"   Duplicate columns: {results_df.columns[results_df.columns.duplicated()].tolist()}")
        # Remove duplicates by keeping first occurrence
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        print(f"   After removing duplicates: {results_df.shape}")
    
    print(f"   Results shape: {results_df.shape}")
    print(f"   CNN features: {len(cnn_feature_columns)}")
    print(f"   TD features: {len(td_df.columns)}")
    print(f"   Total features: {len(cnn_feature_columns) + len(td_df.columns)}")
    
    # Step 7: Save results (Fixed: use subject_id in filename)
    output_file = f'../../data/WESAD_hybrid_features/S{subject_id}_combined_features_both_labels.csv'  # NEW: Dynamic filename
    results_df.to_csv(output_file, index=False)
    print(f"\n7. Saved results to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Subject: S{subject_id}")
    print(f"Data types: Stress (1.0) + Non-stress (0.0)")
    print(f"Total windows: {len(all_cnn_features)}")
    print(f"  - Stress windows: {len(stress_cnn)}")
    print(f"  - Non-stress windows: {len(non_stress_cnn)}")
    print(f"Features per window:")
    print(f"  - CNN features: {len(cnn_feature_columns)}")
    print(f"  - Time-domain features: {len(td_df.columns)}")
    print(f"  - Total features: {len(cnn_feature_columns) + len(td_df.columns)}")
    print(f"Output file: {output_file}")
    
    # Label distribution check - simplified
    print(f"\nFinal label distribution:")
    print(f"  DataFrame shape: {results_df.shape}")
    print(f"  DataFrame columns: {list(results_df.columns[:5])}...")
    
    if 'label' in results_df.columns:
        # Simple manual count
        stress_count = (results_df['label'] == 1.0).sum()
        non_stress_count = (results_df['label'] == 0.0).sum()
        print(f"  Stress (1.0): {stress_count} windows")
        print(f"  Non-stress (0.0): {non_stress_count} windows")
    else:
        print(f"  Warning: 'label' column not found!")
    
    # Show sample features - simplified to avoid pandas issues
    print(f"\nSample features:")
    if len(results_df) > 0:
        print(f"  Results shape: {results_df.shape}")
        print(f"  Column info: {len(results_df.columns)} total columns")
        
        # Just show features from first row to avoid filtering issues
        try:
            print(f"  Subject ID: {results_df.iloc[0]['subject_id']}")  # NEW: Show subject ID
            print(f"  CNN features (first 3): {dict(list(results_df.iloc[0][cnn_feature_columns[:3]].round(4).items()))}")
            print(f"  TD features (first 3): {dict(list(results_df.iloc[0][td_df.columns[:3]].round(4).items()))}")
            print(f"  Label: {results_df.iloc[0]['label']}")
        except Exception as e:
            print(f"  Error showing sample features: {e}")
    else:
        print(f"  No data to show")
    
    return results_df  # NEW: Return the dataframe for potential further use

def run_trained_hybrid_extraction():
    """
    Run the full pipeline with CNN training
    """
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    successful_subjects = []
    failed_subjects = []
    
    for subject in subject_ids:
        print(f"\nProcessing subject S{subject} with trained CNN...")
        try:
            # Use other subjects for CNN training
            other_subjects = [s for s in subject_ids if s != subject][:5]  # Use first 5 others
            
            results = extract_combined_features_with_training(subject, training_subjects=other_subjects)
            if results is not None:
                successful_subjects.append(subject)
                print(f"✓ Successfully processed S{subject}")
            else:
                failed_subjects.append(subject)
                print(f"✗ Failed to process S{subject}")
        except Exception as e:
            failed_subjects.append(subject)
            print(f"✗ Error processing S{subject}: {str(e)}")
    
    print(f"\nFinal results: {len(successful_subjects)} successful, {len(failed_subjects)} failed")
    return successful_subjects, failed_subjects

# Run function to extract features
if __name__ == "__main__":
    print("Starting combined feature extraction for both labels...")
    run_trained_hybrid_extraction()

#%% Combine spreadsheets into one spreadsheet

# combine files into one spreadsheet
# change the label data from 1.0 and 0.0 to '1' and '0'
# save csv

import pandas as pd
import os
import glob

def combine_subject_files():
    """
    Combine all individual subject CSV files into one cohesive spreadsheet
    and convert label values from 1.0/0.0 to 1/0
    """
    
    # Configuration
    DATA_PATH = '../../data/WESAD_hybrid_features/'
    OUTPUT_FILE = '../../data/all_subjects_WESAD_hybrid_features.csv'
    
    # Pattern to match subject files
    file_pattern = os.path.join(DATA_PATH, 'S*_combined_features_both_labels.csv')
    
    print("="*70)
    print("COMBINING ALL SUBJECT FILES")
    print("="*70)
    
    # Find all subject files
    subject_files = glob.glob(file_pattern)
    subject_files.sort()  # Sort to ensure consistent order
    
    if not subject_files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Found {len(subject_files)} subject files:")
    for file in subject_files:
        print(f"  - {os.path.basename(file)}")
    
    # List to store all dataframes
    all_dataframes = []
    successful_subjects = []
    failed_subjects = []
    
    # Process each subject file
    for file_path in subject_files:
        try:
            subject_name = os.path.basename(file_path).split('_')[0]  # Extract S2, S3, etc.
            print(f"\nProcessing {subject_name}...")
            
            # Read the file
            df = pd.read_csv(file_path)
            print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check if label column exists and convert format
            if 'label' in df.columns:
                # Convert 1.0/0.0 to 1/0 (integers)
                df['label'] = df['label'].astype(int)
                
                # Count labels
                label_counts = df['label'].value_counts().sort_index()
                print(f"  Labels: {dict(label_counts)}")
            else:
                print(f"  Warning: No 'label' column found in {subject_name}")
            
            # Add to list
            all_dataframes.append(df)
            successful_subjects.append(subject_name)
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {str(e)}")
            failed_subjects.append(os.path.basename(file_path))
    
    # Combine all dataframes
    if all_dataframes:
        print(f"\nCombining {len(all_dataframes)} dataframes...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"Combined dataset shape: {combined_df.shape}")
        
        # Summary statistics
        if 'label' in combined_df.columns:
            print(f"\nLabel distribution in combined dataset:")
            label_summary = combined_df['label'].value_counts().sort_index()
            for label, count in label_summary.items():
                percentage = (count / len(combined_df)) * 100
                print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
        
        if 'subject_id' in combined_df.columns:
            print(f"\nSubject distribution:")
            subject_summary = combined_df['subject_id'].value_counts().sort_index()
            for subject, count in subject_summary.items():
                print(f"  S{subject}: {count} samples")
        
        # Save combined file
        print(f"\nSaving combined dataset to: {OUTPUT_FILE}")
        combined_df.to_csv(OUTPUT_FILE, index=False)
        
        # Final summary
        print("\n" + "="*70)
        print("COMBINATION SUMMARY")
        print("="*70)
        print(f"Total subjects processed: {len(subject_files)}")
        print(f"Successfully combined: {len(successful_subjects)} - {successful_subjects}")
        if failed_subjects:
            print(f"Failed: {len(failed_subjects)} - {failed_subjects}")
        print(f"Combined dataset: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
        print(f"Output file: {OUTPUT_FILE}")
        
        return combined_df
    
    else:
        print("No dataframes to combine!")
        return None

# Run the combination
if __name__ == "__main__":
    # Combine all subject files
    combined_data = combine_subject_files()
# %%
