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

def extract_combined_features_s2_both_labels():
    """
    Extract CNN and time-domain features from S2 data for both stress (1.0) and non-stress (0.0)
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
    print("COMBINED CNN + TD FEATURES: S2 STRESS (1.0) + NON-STRESS (0.0)")
    print("="*70)
    print(f"Window size: {WINDOW_SIZE}s ({WINDOW_SAMPLES} samples)")
    print(f"Step size: {STEP_SIZE}s ({STEP_SAMPLES} samples)")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    
    # Step 1: Load S2 data
    print("\n1. Loading S2 data...")
    s2_file = os.path.join(DATA_PATH, 'S2.csv')
    
    if not os.path.exists(s2_file):
        print(f"Error: File not found at {s2_file}")
        return None
    
    df = pd.read_csv(s2_file)
    print(f"   Loaded S2 data: {len(df)} samples")
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
    
    # Create labels dataframe
    labels_df = pd.DataFrame({'label': all_labels})
    
    # Combine: Labels + CNN Features + TD Features
    results_df = pd.concat([labels_df, cnn_df, td_df], axis=1)
    
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
    
    # Step 7: Save results
    output_file = '../../data/S2_combined_features_both_labels.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n7. Saved results to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Subject: S2")
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
            print(f"  CNN features (first 3): {dict(list(results_df.iloc[0][cnn_feature_columns[:3]].round(4).items()))}")
            print(f"  TD features (first 3): {dict(list(results_df.iloc[0][td_df.columns[:3]].round(4).items()))}")
            print(f"  Label: {results_df.iloc[0]['label']}")
        except Exception as e:
            print(f"  Error showing sample features: {e}")
    else:
        print(f"  No data to show")
    
    # Feature comparison summary
    print(f"\nREADY FOR REVIEWER COMPARISON:")
    print(f"  1. Traditional ML: {len(td_df.columns)} handcrafted features")
    print(f"  2. Deep Learning: {len(cnn_feature_columns)} CNN features") 
    print(f"  3. Hybrid: {len(cnn_feature_columns) + len(td_df.columns)} combined features")
    print(f"  Computational ratio: {len(cnn_feature_columns) / len(td_df.columns):.1f}x more features with CNN")
    
    return results_df

# Example usage
if __name__ == "__main__":
    print("Starting combined feature extraction for both labels...")
    
    # Extract features for both stress and non-stress
    results = extract_combined_features_s2_both_labels()
    
    if results is not None:
        print("\n" + "="*70)
        print("✅ FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutput file ready for three-way comparison:")
        print("1. Traditional ML: Use 'label' + TD feature columns only")
        print("2. Deep Learning: Use 'label' + CNN feature columns only") 
        print("3. Hybrid Approach: Use 'label' + all feature columns")
        print("\nPerfect for addressing reviewer's mainstream method concerns!")
    else:
        print("\n❌ Feature extraction failed. Please check the error messages above.")