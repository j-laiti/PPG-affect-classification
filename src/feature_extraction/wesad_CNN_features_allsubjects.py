import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter

class Simple1DCNNFeatures(nn.Module):
    """
    Simple CNN for feature extraction from PPG signals
    Based on Zhao et al. temporal convolutional approach
    """
    def __init__(self, input_length=7680):  # 120s at 64Hz
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection (kernel=15, ~0.23s)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Layer 2: Pattern refinement (kernel=11, ~0.17s)  
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Layer 3: High-level features (kernel=7, ~0.11s)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # Fixed output: 32 x 4 = 128 features
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)  # Output: (batch_size, 128)
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
                window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)
                window_tensor = window_tensor.to(device)
                
                cnn_features = model(window_tensor)
                cnn_features_np = cnn_features.cpu().numpy().flatten()
                
                # ===== TIME-DOMAIN FEATURE EXTRACTION =====
                bp_bvp = bandpass_filter(window, 0.2, 10, fs_dict['BVP'], order=2)
                smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
                
                td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                          fs=fs_dict['BVP'], 
                                          label=int(label_value), 
                                          calc_sq=False)
                
                # Check if both extractions were successful
                if td_stats and len(td_stats) > 0:
                    all_cnn_features.append(cnn_features_np)
                    all_td_features.append(td_stats)
                    all_labels.append(label_value)
                    
                    if (i + 1) % 20 == 0 or (i + 1) == len(windows):
                        print(f"     Processed {i+1}/{len(windows)} windows ({len(all_cnn_features)} successful)")
                        
            except Exception as e:
                print(f"     Error processing window {i}: {str(e)}")
                continue
    
    print(f"   Successfully extracted features from {len(all_cnn_features)}/{len(windows)} windows")
    return all_cnn_features, all_td_features, all_labels

def extract_combined_features_both_labels(subject_id):
    """
    Extract CNN and time-domain features from all subject data for both stress (1.0) and non-stress (0.0)
    Outputs combined features to single CSV file
    """
    
    # Configuration
    DATA_PATH = '../../data/WESAD_BVP_extracted/'
    WINDOW_SIZE = 120  # seconds
    STEP_SIZE = 30     # seconds (sliding window step)
    SAMPLING_RATE = 64 # Hz
    
    WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE  # 7680 samples
    STEP_SAMPLES = STEP_SIZE * SAMPLING_RATE      # 1920 samples
    
    print("="*70)
    print(f"COMBINED CNN + TD FEATURES: S{subject_id} STRESS (1.0) + NON-STRESS (0.0)")  # Fixed: use subject_id
    print("="*70)
    print(f"Window size: {WINDOW_SIZE}s ({WINDOW_SAMPLES} samples)")
    print(f"Step size: {STEP_SIZE}s ({STEP_SAMPLES} samples)")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    
    # Step 1: Load subject data (Fixed: use subject_id parameter)
    print(f"\n1. Loading S{subject_id} data...")
    subject_file = os.path.join(DATA_PATH, f'S{subject_id}.csv')

    if not os.path.exists(subject_file):
        print(f"Error: File not found at {subject_file}")
        return None
    
    df = pd.read_csv(subject_file)
    print(f"   Loaded data: {len(df)} samples")
    print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Step 2: Separate stress and non-stress data
    print("\n2. Separating data by labels...")
    stress_data = df[df['label'] == 1.0].copy()
    non_stress_data = df[df['label'] == 0.0].copy()
    
    print(f"   Stress (1.0): {len(stress_data)} samples ({len(stress_data)/SAMPLING_RATE:.1f} seconds)")
    print(f"   Non-stress (0.0): {len(non_stress_data)} samples ({len(non_stress_data)/SAMPLING_RATE:.1f} seconds)")
    
    # Step 3: Initialize CNN model
    print("\n3. Initializing CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = Simple1DCNNFeatures(input_length=WINDOW_SAMPLES)
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Step 4: Extract features for both labels
    print("\n4. Extracting features for both labels...")
    fs_dict = {'BVP': SAMPLING_RATE}
    
    # Extract stress features
    stress_cnn, stress_td, stress_labels = extract_features_for_label(
        stress_data['BVP'].values, 1.0, model, device, fs_dict, WINDOW_SAMPLES, STEP_SAMPLES
    )
    
    # Extract non-stress features  
    non_stress_cnn, non_stress_td, non_stress_labels = extract_features_for_label(
        non_stress_data['BVP'].values, 0.0, model, device, fs_dict, WINDOW_SAMPLES, STEP_SAMPLES
    )
    
    # Step 5: Combine all features
    print("\n5. Combining all features...")
    
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
    
    # Step 6: Create combined dataframe
    print("\n6. Creating combined dataframe...")
    
    # Create subject ID column (NEW: Add subject ID for each row)
    subject_ids = [subject_id] * len(all_cnn_features)
    
    # Create CNN features dataframe
    cnn_feature_columns = [f'cnn_feature_{i:04d}' for i in range(128)]
    cnn_df = pd.DataFrame(all_cnn_features, columns=cnn_feature_columns)
    
    # Create TD features dataframe
    if isinstance(all_td_features[0], dict):
        td_df = pd.DataFrame(all_td_features)
    else:
        # If it returns a list, define column names (adjust based on your actual features)
        td_feature_columns = [
            'mean_hr', 'std_hr', 'rmssd', 'sdnn', 'sdsd', 
            'mean_nn', 'mean_sd', 'median_nn', 'pnn20', 'pnn50'
        ]
        td_df = pd.DataFrame(all_td_features, columns=td_feature_columns[:len(all_td_features[0])])
    
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
    output_file = f'../../data/S{subject_id}_combined_features_both_labels.csv'  # NEW: Dynamic filename
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

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Run function to extract features
if __name__ == "__main__":
    print("Starting combined feature extraction for both labels...")
    
    # Extract features for both stress and non-stress
    successful_subjects = []
    failed_subjects = []
    
    for subject in subject_ids:
        print(f"\nProcessing subject S{subject}...")
        try:
            results = extract_combined_features_both_labels(subject)
            if results is not None:
                successful_subjects.append(subject)
                print(f"✓ Successfully processed S{subject}")
            else:
                failed_subjects.append(subject)
                print(f"✗ Failed to process S{subject}")
        except Exception as e:
            failed_subjects.append(subject)
            print(f"✗ Error processing S{subject}: {str(e)}")
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total subjects attempted: {len(subject_ids)}")
    print(f"Successful: {len(successful_subjects)} - {successful_subjects}")
    print(f"Failed: {len(failed_subjects)} - {failed_subjects}")

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
    DATA_PATH = '../../data/WESAD_CNN_TD_combined_features/'
    OUTPUT_FILE = '../../data/all_subjects_CNN_and_TD_features.csv'
    
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

def verify_combined_file(file_path='../../data/combined_all_subjects_features.csv'):
    """
    Verify the combined file and show summary statistics
    """
    if not os.path.exists(file_path):
        print(f"Combined file not found: {file_path}")
        return
    
    print("\n" + "="*70)
    print("VERIFICATION OF COMBINED FILE")
    print("="*70)
    
    df = pd.read_csv(file_path)
    
    print(f"File: {file_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show first few column names
    print(f"\nFirst 10 columns: {list(df.columns[:10])}")
    
    # Label verification
    if 'label' in df.columns:
        print(f"\nLabel column type: {df['label'].dtype}")
        print(f"Label values: {sorted(df['label'].unique())}")
        print(f"Label distribution:")
        for label, count in df['label'].value_counts().sort_index().items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Subject verification
    if 'subject_id' in df.columns:
        print(f"\nSubject IDs: {sorted(df['subject_id'].unique())}")
        print(f"Samples per subject:")
        for subject, count in df['subject_id'].value_counts().sort_index().items():
            print(f"  S{subject}: {count} samples")
    
    # Show sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

# Run the combination
if __name__ == "__main__":
    # Combine all subject files
    combined_data = combine_subject_files()
    
    if combined_data is not None:
        # Verify the results
        verify_combined_file()
        print("\n✓ File combination completed successfully!")
    else:
        print("\n✗ File combination failed!")
# %%
