"""
AKTIVES PPG Feature Extraction

Extracts HRV features from all AKTIVES analysis windows using the lightweight
PPG processing pipeline. Processes windows from all cohorts and compiles results.

Usage:
    python aktives_td_ppg_extraction.py
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from preprocessing.filters import *
from preprocessing.peak_detection import *
from preprocessing.feature_extraction import *

#%% Configuration
FS = 64  # Sampling frequency in Hz

COHORT_PATHS = {
    'TD': 'Typically Developed',
    'dyslexia': 'Dyslexia',
    'ID': 'Intellectual Disabilities',
    'OBPI': 'Obstetric Brachial Plexus Injuries'
}

#%% Feature Extraction Functions

def process_single_window(row, cohort_folder, fs=64):
    """
    Extract HRV features from a single analysis window.
    
    Args:
        row: Window row from analysis_windows CSV
        cohort_folder: Name of cohort folder (e.g., "Typically Developed")
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of extracted features or None if processing failed
    """
    try:
        # Extract window information
        participant_cell = row["Participant"]
        participant = participant_cell.split("_")[0]
        game = participant_cell.split("_")[1]
        
        interval_start = row["Interval_Start"]
        interval_end = row["Interval_End"]
        label = row["Label"]
        
        # Load PPG data
        ppg_file_path = f"../../data/Aktives/PPG/{cohort_folder}/{participant}/{game}/BVP.csv" # TODO: Update path based on your setup
        
        if not os.path.exists(ppg_file_path):
            return {'error': f"PPG file not found: {ppg_file_path}"}
        
        relevant_ppg_data = pd.read_csv(ppg_file_path)
        
        # Add time column and clean values
        relevant_ppg_data["Time"] = relevant_ppg_data.index / fs
        relevant_ppg_data['values'] = (
            relevant_ppg_data['values']
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
        
        # Extract interval
        ppg_interval = relevant_ppg_data[
            (relevant_ppg_data['Time'] >= interval_start) & 
            (relevant_ppg_data['Time'] <= interval_end)
        ]
        
        if len(ppg_interval) == 0:
            return {'error': f"No data in interval {interval_start}-{interval_end}"}
        
        raw_ppg_values = ppg_interval['values'].values
        
        # Check for sufficient data
        if len(raw_ppg_values) < fs * 10:
            print(f"WARNING: Only {len(raw_ppg_values)/fs:.2f} seconds of data")
        
        # Process PPG signal
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        if len(clean_ppg_values) == 0:
            return {'error': "All PPG values are NaN"}
        
        ppg_standardized = standardize(clean_ppg_values)
        
        # Apply filtering pipeline
        bandpass_signal = bandpass_filter(
            ppg_standardized, 
            lowcut=0.5, 
            highcut=10.0, 
            fs=fs, 
            order=2
        )
        smoothed_signal = moving_average_filter(bandpass_signal, window_size=5)
        
        # Noise elimination
        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, fs, 95)
        clean_signal, clean_indices = simple_noise_elimination(
            smoothed_signal, 
            fs, 
            std_ths
        )
        
        final_clean_signal = moving_average_filter(clean_signal, window_size=3)
        
        # Check clean signal quality
        if len(final_clean_signal) < fs * 5:
            print(f"WARNING: Only {len(final_clean_signal)/fs:.2f} seconds of clean data")
        
        # Peak detection
        peaks_threshold = threshold_peakdetection(final_clean_signal, fs)
        
        if len(peaks_threshold) < 3:
            print(f"WARNING: Only {len(peaks_threshold)} peaks detected")
        
        # Extract features
        ppg_features = get_ppg_features(
            final_clean_signal, 
            fs, 
            label, 
            ppg_standardized, 
            calc_sq=True
        )
        
        # Add metadata
        ppg_features.update({
            'participant': participant,
            'game': game,
            'interval_start': interval_start,
            'interval_end': interval_end,
            'raw_signal_length': len(raw_ppg_values),
            'clean_signal_length': len(final_clean_signal),
            'num_peaks': len(peaks_threshold)
        })
        
        return ppg_features
        
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}


def process_cohort(cohort_name, cohort_folder):
    """
    Process all analysis windows for a single cohort.
    
    Args:
        cohort_name: Short cohort identifier (e.g., 'TD', 'dyslexia')
        cohort_folder: Full cohort folder name
        
    Returns:
        Tuple of (features_df, errors_list)
    """
    print(f"\n{'='*60}")
    print(f"Processing {cohort_name} cohort")
    print(f"{'='*60}")
    
    # Load analysis windows
    windows_file = f"../../data/Aktives/analysis_windows/analysis_windows_{cohort_name}.csv" # TODO: Update path based on your setup
    
    if not os.path.exists(windows_file):
        print(f"ERROR: Analysis windows file not found: {windows_file}")
        return None, None
    
    windows = pd.read_csv(windows_file)
    print(f"Found {len(windows)} analysis windows to process")
    
    all_features = []
    processing_errors = []
    
    # Process each window
    for idx, row in tqdm(windows.iterrows(), total=len(windows), desc="Processing windows"):
        result = process_single_window(row, cohort_folder, fs=FS)
        
        if result is not None and 'error' not in result:
            result['window_idx'] = idx
            result['cohort'] = cohort_name
            all_features.append(result)
        else:
            error_info = {
                'window_idx': idx,
                'cohort': cohort_name,
                'error': result.get('error', 'Unknown error') if result else 'No result'
            }
            processing_errors.append(error_info)
    
    # Create results DataFrame
    if len(all_features) > 0:
        features_df = pd.DataFrame(all_features)
        
        # Save cohort features
        output_file = f"../../data/Aktives/extracted_features/ppg_features_{cohort_name}.csv" # TODO: Update path based on your setup
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        features_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully processed: {len(all_features)}/{len(windows)} windows")
        print(f"✓ Features saved to: {output_file}")
        
        return features_df, processing_errors
    else:
        print(f"\nERROR: No features extracted for {cohort_name}")
        return None, processing_errors


def compile_all_cohorts():
    """
    Compile extracted features from all cohorts into a single dataset.
    
    Returns:
        DataFrame: Compiled AKTIVES PPG features
    """
    print(f"\n{'='*60}")
    print("Compiling all cohort features")
    print(f"{'='*60}")
    
    feature_files = [
        f"../../data/Aktives/extracted_features/ppg_features_{cohort}.csv" # TODO: Update path based on your setup
        for cohort in COHORT_PATHS.keys()
    ]
    
    df_list = []
    for file in feature_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"✓ Loaded {len(df)} features from {os.path.basename(file)}")
        else:
            print(f"⚠ Feature file not found: {file}")
    
    if len(df_list) == 0:
        print("ERROR: No feature files found to compile")
        return pd.DataFrame()
    
    compiled_df = pd.concat(df_list, ignore_index=True)
    output_file = "../../data/Aktives/extracted_features/AKTIVES_ppg_features_merged.csv" # TODO: Update path based on your setup
    compiled_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Compiled {len(compiled_df)} total features")
    print(f"✓ Saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"  Unique participants: {compiled_df['participant'].nunique()}")
    print(f"  Label distribution:")
    print(compiled_df['label'].value_counts())
    print(f"\n  Cohort distribution:")
    print(compiled_df['cohort'].value_counts())
    
    return compiled_df

#%% Main Execution

if __name__ == "__main__":
    all_errors = []
    
    # Process each cohort
    for cohort_name, cohort_folder in COHORT_PATHS.items():
        features_df, errors = process_cohort(cohort_name, cohort_folder)
        
        if errors:
            all_errors.extend(errors)
    
    # Save error log if any errors occurred
    if len(all_errors) > 0:
        errors_df = pd.DataFrame(all_errors)
        error_file = "../../data/Aktives/extracted_features/processing_errors.csv" # TODO: Update path based on your setup
        errors_df.to_csv(error_file, index=False)
        
        print(f"\n⚠ Total errors: {len(all_errors)}")
        print(f"⚠ Error log saved to: {error_file}")
        print(f"\nMost common errors:")
        print(errors_df['error'].value_counts().head())
    
    # Compile all cohorts
    compiled_df = compile_all_cohorts()
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")