"""
AKTIVES PPG Data Compilation Script

Processes the AKTIVES dataset for PPG signal analysis.
Compiles all PPG windows from different cohorts into a single spreadsheet.

Simple format: each column is one window with identifier as header and PPG data below.
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""


#%% code setup and necessary functions
# imports
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

def load_all_aktives_windows():
    """
    Load all analysis windows from all cohorts
    
    Returns:
        combined_windows: DataFrame with all windows and cohort info
    """
    print("Loading all AKTIVES analysis windows...")
    
    cohorts = ['dyslexia', 'ID', 'OBPI', 'TD']
    all_windows = []
    
    for cohort in cohorts:
        window_file = f"../../data/Aktives/analysis_windows/analysis_windows_{cohort}.csv" # TODO: Update path based on your setup
        
        if os.path.exists(window_file):
            windows_df = pd.read_csv(window_file)
            windows_df['cohort'] = cohort
            all_windows.append(windows_df)
            print(f"  Loaded {cohort}: {len(windows_df)} windows")
        else:
            print(f"  Warning: {window_file} not found")
    
    if not all_windows:
        raise ValueError("No analysis windows found!")
    
    combined_windows = pd.concat(all_windows, ignore_index=True)
    print(f"Total windows loaded: {len(combined_windows)}")
    
    return combined_windows

def extract_ppg_window_data(row, fs=64, max_samples=1920):
    """
    Extract PPG data for a single AKTIVES window
    
    Args:
        row: Window row from analysis_windows CSV
        fs: Sampling frequency (Hz)
        max_samples: Maximum samples (30 seconds * 64 Hz = 1920)
    
    Returns:
        tuple: (column_identifier, ppg_values_array)
    """
    try:
        # Extract window information
        participant_cell = row["Participant"]
        participant = participant_cell.split("_")[0]
        game = participant_cell.split("_")[1]
        
        interval_start = row["Interval_Start"]
        interval_end = row["Interval_End"]
        cohort = row["cohort"]
        
        # Create column identifier
        column_id = f"{participant}_{game}_{interval_start}"
        
        # Map cohort names to folder names
        cohort_folder_map = {
            'dyslexia': 'Dyslexia',
            'ID': 'Intellectual Disabilities',
            'OBPI': 'Obstetric Brachial Plexus Injuries',
            'TD': 'Typically Developed'
        }
        
        cohort_folder = cohort_folder_map[cohort]
        
        # Load PPG data
        ppg_file_path = f"../../data/Aktives/PPG/{cohort_folder}/{participant}/{game}/BVP.csv"
        
        if not os.path.exists(ppg_file_path):
            print(f"Warning: PPG file not found: {ppg_file_path}")
            return column_id, None
        
        ppg_data = pd.read_csv(ppg_file_path)
        
        # Add time column
        ppg_data["Time"] = ppg_data.index / fs
        
        # Clean and convert values
        ppg_data['values'] = ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # Extract window data
        window_mask = (ppg_data["Time"] >= interval_start) & (ppg_data["Time"] <= interval_end)
        window_ppg = ppg_data[window_mask]['values'].values
        
        if len(window_ppg) == 0:
            print(f"Warning: No PPG data found for window {column_id}")
            return column_id, None
        
        # Limit to max_samples (30 seconds worth)
        if len(window_ppg) > max_samples:
            window_ppg = window_ppg[:max_samples]
        
        return column_id, window_ppg
        
    except Exception as e:
        print(f"Error processing window: {e}")
        return f"error_{row['Participant'] if 'Participant' in row else 'unknown'}", None

def compile_aktives_ppg_simple(output_path="aktives_ppg_simple.csv", max_seconds=30):
    """
    Compile all AKTIVES PPG windows into a simple spreadsheet format
    Each column is one window with identifier as header and PPG data below
    
    Args:
        output_path: Path for output CSV file
        max_seconds: Maximum seconds of data per window (default 30)
    
    Returns:
        DataFrame: Compiled PPG data in simple format
    """
    fs = 64
    max_samples = max_seconds * fs  # 30 seconds * 64 Hz = 1920 samples
    
    # Load all windows
    windows_df = load_all_aktives_windows()
    
    print(f"Extracting PPG data for all windows (max {max_seconds} seconds per window)...")
    
    # Dictionary to store all columns
    all_columns = {}
    
    # Process each window
    for idx, row in tqdm(windows_df.iterrows(), total=len(windows_df), desc="Processing windows"):
        column_id, ppg_data = extract_ppg_window_data(row, fs=fs, max_samples=max_samples)
        
        if ppg_data is not None and len(ppg_data) > 0:
            all_columns[column_id] = ppg_data
    
    if not all_columns:
        raise ValueError("No valid PPG windows extracted!")
    
    print(f"Successfully extracted {len(all_columns)} windows")
    
    # Create DataFrame where each column is a window
    # Pad shorter columns with NaN to match the longest
    max_length = max(len(data) for data in all_columns.values())
    print(f"Maximum window length: {max_length} samples ({max_length/fs:.1f} seconds)")
    
    # Pad all columns to same length
    padded_columns = {}
    for col_id, data in all_columns.items():
        if len(data) < max_length:
            # Pad with NaN
            padded_data = np.full(max_length, np.nan)
            padded_data[:len(data)] = data
            padded_columns[col_id] = padded_data
        else:
            padded_columns[col_id] = data
    
    # Create DataFrame
    compiled_df = pd.DataFrame(padded_columns)
    
    # Save to CSV
    print(f"Saving compiled data to: {output_path}")
    compiled_df.to_csv(output_path, index=False)
    print(f"Saved {len(compiled_df)} rows x {len(compiled_df.columns)} columns to {output_path}")
    
    return compiled_df

#%% extract relevant AKTIVES data
if __name__ == "__main__":
    # Compile PPG data in simple format (30 seconds max per window)
    df_simple = compile_aktives_ppg_simple("aktives_ppg_simple.csv", max_seconds=30)
    
    print("\nFirst few columns and rows:")
    print(df_simple.iloc[:10, :5])  # Show first 10 rows, 5 columns
    print(f"\nTotal shape: {df_simple.shape}")
    print(f"Column names (first 5): {list(df_simple.columns[:5])}")