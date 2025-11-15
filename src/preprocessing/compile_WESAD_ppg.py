"""
WESAD BVP Signal Extraction

Extracts raw BVP signals and labels from WESAD pickle files and saves them
as time-aligned CSV files for easier access and analysis.

Usage:
    python compile_WESAD_ppg.py
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import os
import pickle
import pandas as pd

#%% Configuration
WESAD_PATH = '../../data/WESAD/' # TODO: Update path based on your setup
OUTPUT_PATH = '../../data/WESAD_BVP_extracted/'

# Sampling frequencies from WESAD
FS_BVP = 64  # Hz
FS_LABEL = 700  # Hz

# Label mapping: baseline=0, stress=1, amusement=0
LABEL_MAPPING = {
    1: 0,  # baseline -> 0
    2: 1,  # stress -> 1  
    3: 0   # amusement -> 0
}

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

#%% Extraction Functions

def extract_bvp_for_subject(subject_id):
    """
    Extract BVP signal and labels for one subject and save to CSV.
    
    Args:
        subject_id: WESAD subject ID number
        
    Returns:
        Tuple of (total_samples, baseline_count, stress_count)
    """
    subject_name = f'S{subject_id}'
    pickle_path = os.path.join(WESAD_PATH, subject_name, f'{subject_name}.pkl')
    
    print(f"Loading {subject_name}...")
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    
    # Extract BVP signal from wrist and labels
    bvp_signal = data['signal']['wrist']['BVP']
    labels = data['label']
    
    print(f"  BVP samples: {len(bvp_signal)} at {FS_BVP}Hz")
    print(f"  Label samples: {len(labels)} at {FS_LABEL}Hz")
    
    # Create time-aligned dataframes
    bvp_df = pd.DataFrame(bvp_signal, columns=['BVP'])
    bvp_df.index = [(1 / FS_BVP) * i for i in range(len(bvp_df))]
    
    label_df = pd.DataFrame(labels, columns=['label'])
    label_df.index = [(1 / FS_LABEL) * i for i in range(len(label_df))]
    
    # Convert to datetime for joining
    bvp_df.index = pd.to_datetime(bvp_df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')
    
    # Join BVP with labels
    combined_df = bvp_df.join(label_df, how='outer')
    
    # Forward and backward fill labels
    combined_df['label'] = combined_df['label'].ffill().bfill()
    
    # Remove any remaining NaN rows
    combined_df = combined_df.dropna()
    
    # Reset index and map labels to binary
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['label'] = combined_df['label'].map(LABEL_MAPPING)
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_PATH, f'{subject_name}.csv')
    combined_df.to_csv(output_file, index=False)
    
    # Calculate summary statistics
    baseline_count = (combined_df['label'] == 0).sum()
    stress_count = (combined_df['label'] == 1).sum()
    duration_min = len(combined_df) / FS_BVP / 60
    
    print(f"  Saved {len(combined_df)} samples ({duration_min:.1f} minutes)")
    print(f"  Baseline: {baseline_count}, Stress: {stress_count}")
    
    return len(combined_df), baseline_count, stress_count

#%% Main Execution

def main():
    """Extract BVP data for all subjects."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("Extracting BVP data from WESAD...")
    print(f"Input: {WESAD_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("-" * 50)
    
    total_samples = 0
    total_baseline = 0
    total_stress = 0
    
    for subject_id in SUBJECT_IDS:
        try:
            samples, baseline, stress = extract_bvp_for_subject(subject_id)
            total_samples += samples
            total_baseline += baseline
            total_stress += stress
            print()
        except Exception as e:
            print(f"  Error with S{subject_id}: {e}")
            print()
    
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print(f"Total samples: {total_samples:,}")
    print(f"Baseline samples: {total_baseline:,}")
    print(f"Stress samples: {total_stress:,}")
    print(f"Files saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

#%% Verification Cell - Load and inspect extracted data
if __name__ == "__main__":
    df = pd.read_csv('../../data/WESAD_BVP_extracted/S2.csv')
    
    print("Shape:", df.shape)
    print("\nLabel value counts:")
    print(df['label'].value_counts(dropna=False))
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nAny NaN values?")
    print(df.isnull().sum())