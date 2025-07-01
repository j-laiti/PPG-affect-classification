# Automated PPG feature extraction for all analysis windows
# Based on your original single-window processing code

#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # for progress bar

from preprocessing.filters import *
from preprocessing.peak_detection import *
from preprocessing.feature_extraction import *

# Load analysis windows
windows = pd.read_csv("../../data/Aktives/analysis_windows/analysis_windows_ID.csv")
print(f"Found {len(windows)} analysis windows to process")

# Initialize lists to store results
all_features = []
processing_errors = []
fs = 64  # Sampling frequency in Hz

#%% Process each window
for idx, row in tqdm(windows.iterrows(), total=len(windows), desc="Processing windows"):
    try:
        # Extract window information
        participant_cell = row["Participant"]
        participant = participant_cell.split("_")[0]
        game = participant_cell.split("_")[1]
        
        interval_start = row["Interval_Start"]
        interval_end = row["Interval_End"]
        label = row["Label"]
        
        print(f"\n--- Processing Window {idx+1}/{len(windows)} ---")
        print(f"Participant: {participant}, Game: {game}")
        print(f"Interval: {interval_start} to {interval_end} seconds")
        print(f"Label: {label}")
        
        # Load PPG data for this participant and game
        ppg_file_path = f"../../data/Aktives/PPG/Intellectual Disabilities/{participant}/{game}/BVP.csv"
        
        # Check if file exists
        if not os.path.exists(ppg_file_path):
            error_msg = f"PPG file not found: {ppg_file_path}"
            print(f"ERROR: {error_msg}")
            processing_errors.append({
                'window_idx': idx,
                'participant': participant,
                'game': game,
                'error': error_msg
            })
            continue
        
        # Load PPG data
        relevant_ppg_data = pd.read_csv(ppg_file_path)
        
        # Add time column (in seconds from zero)
        relevant_ppg_data["Time"] = relevant_ppg_data.index / fs
        
        # Clean and convert values
        relevant_ppg_data['values'] = relevant_ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # Select the interval
        ppg_interval = relevant_ppg_data[(relevant_ppg_data['Time'] >= interval_start) & 
                                       (relevant_ppg_data['Time'] <= interval_end)]
        
        if len(ppg_interval) == 0:
            error_msg = f"No data found in interval {interval_start} to {interval_end}"
            print(f"ERROR: {error_msg}")
            processing_errors.append({
                'window_idx': idx,
                'participant': participant,
                'game': game,
                'error': error_msg
            })
            continue
        
        raw_ppg_values = ppg_interval['values'].values
        
        # Check for sufficient data
        if len(raw_ppg_values) < fs * 10:  # Less than 10 seconds of data
            error_msg = f"Insufficient data: only {len(raw_ppg_values)} samples ({len(raw_ppg_values)/fs:.2f} seconds)"
            print(f"WARNING: {error_msg}")
        
        # Process the PPG signal
        
        # Remove NaN and standardize
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        if len(clean_ppg_values) == 0:
            error_msg = "All PPG values are NaN"
            print(f"ERROR: {error_msg}")
            processing_errors.append({
                'window_idx': idx,
                'participant': participant,
                'game': game,
                'error': error_msg
            })
            continue
            
        ppg_standardized = standardize(clean_ppg_values)
        
        # Apply filters step by step
        # Step 1: Bandpass filter
        bandpass_signal = bandpass_filter(ppg_standardized, lowcut=0.5, highcut=10.0, fs=fs, order=2)
        
        # Step 2: Moving average smoothing
        smoothed_signal = moving_average_filter(bandpass_signal, window_size=5)
        
        # Step 3: Simple dynamic thresholding and noise elimination
        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 95)
        clean_signal, clean_indices = simple_noise_elimination(smoothed_signal, fs, std_ths)
        
        # Step 4: Final smoothing
        final_clean_signal = moving_average_filter(clean_signal, window_size=3)
        
        # Check if we have enough clean signal
        if len(final_clean_signal) < fs * 5:  # Less than 5 seconds of clean data
            error_msg = f"Insufficient clean data: only {len(final_clean_signal)} samples ({len(final_clean_signal)/fs:.2f} seconds)"
            print(f"WARNING: {error_msg}")
        
        # Run peak detection
        peaks_threshold = threshold_peakdetection(final_clean_signal, fs)
        
        if len(peaks_threshold) < 3:  # Need at least 3 peaks for HRV
            error_msg = f"Insufficient peaks detected: only {len(peaks_threshold)} peaks"
            print(f"WARNING: {error_msg}")
        
        # Extract PPG features
        ppg_features = get_ppg_features(final_clean_signal, fs, label, ppg_standardized, calc_sq=True)
        
        # Add metadata to features
        ppg_features['window_idx'] = idx
        ppg_features['participant'] = participant
        ppg_features['game'] = game
        ppg_features['interval_start'] = interval_start
        ppg_features['interval_end'] = interval_end
        ppg_features['raw_signal_length'] = len(raw_ppg_values)
        ppg_features['clean_signal_length'] = len(final_clean_signal)
        ppg_features['num_peaks'] = len(peaks_threshold)
        ppg_features['cohort'] = "ID"
        
        all_features.append(ppg_features)
        print(f"✓ Successfully processed window {idx+1}")
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        processing_errors.append({
            'window_idx': idx,
            'participant': participant if 'participant' in locals() else 'unknown',
            'game': game if 'game' in locals() else 'unknown',
            'error': error_msg
        })
        continue

# Compile results
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total windows: {len(windows)}")
print(f"Successfully processed: {len(all_features)}")
print(f"Errors encountered: {len(processing_errors)}")

if len(all_features) > 0:
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save results
    output_file = "../../data/Aktives/extracted_features/ppg_features_ID.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features_df.to_csv(output_file, index=False)
    print(f"✓ Features saved to: {output_file}")
    
    # Display summary statistics
    print(f"\n=== FEATURE EXTRACTION SUMMARY ===")
    print(f"Total features extracted: {len(features_df)}")
    print(f"Unique participants: {features_df['participant'].nunique()}")
    print(f"Unique games: {features_df['game'].nunique()}")
    print(f"Label distribution:")
    print(features_df['label'].value_counts())
    
    # Signal quality summary
    if 'clean_signal_length' in features_df.columns:
        print(f"\nSignal length statistics (samples):")
        print(f"  Raw signal - Mean: {features_df['raw_signal_length'].mean():.1f}, "
              f"Std: {features_df['raw_signal_length'].std():.1f}")
        print(f"  Clean signal - Mean: {features_df['clean_signal_length'].mean():.1f}, "
              f"Std: {features_df['clean_signal_length'].std():.1f}")
        print(f"  Peak count - Mean: {features_df['num_peaks'].mean():.1f}, "
              f"Std: {features_df['num_peaks'].std():.1f}")

if len(processing_errors) > 0:
    # Save error log
    errors_df = pd.DataFrame(processing_errors)
    error_file = "../../data/Aktives/extracted_features/processing_errors.csv"
    errors_df.to_csv(error_file, index=False)
    print(f"⚠ Error log saved to: {error_file}")
    
    # Display error summary
    print(f"\n=== ERROR SUMMARY ===")
    print("Most common errors:")
    print(errors_df['error'].value_counts().head())

print("\n=== PROCESSING COMPLETE ===")
# %%
