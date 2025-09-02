#%%
import numpy as np
import pandas as pd
import sys
import os
import time
import glob

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from preprocessing.feature_extraction import *
from preprocessing.filters import *

def extract_td_features_for_subject(subject_id):
    """Extract time-domain features for one subject"""
    DATA_PATH = '../../data/WESAD_BVP_extracted/'
    WINDOW_SAMPLES = 120 * 64  # 120s at 64Hz
    STEP_SAMPLES = 30 * 64     # 30s step
    
    print(f"Extracting TD features for S{subject_id}...")
    
    # Load subject data
    subject_file = os.path.join(DATA_PATH, f'S{subject_id}.csv')
    if not os.path.exists(subject_file):
        print(f"File not found: {subject_file}")
        return None
    
    df = pd.read_csv(subject_file)
    
    # Extract features for both labels
    all_td_features = []
    all_labels = []
    all_subject_ids = []
    
    for label_value in [0.0, 1.0]:
        label_data = df[df['label'] == label_value]['BVP'].values
        
        if len(label_data) < WINDOW_SAMPLES:
            print(f"   Warning: Not enough data for label {label_value}")
            continue
        
        # Create windows and extract TD features
        for start_idx in range(0, len(label_data) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = label_data[start_idx:start_idx + WINDOW_SAMPLES]
            
            try:
                # Apply preprocessing pipeline (same as hybrid approach)
                bp_bvp = bandpass_filter(window, 0.2, 10, 64, order=2)
                smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
                
                segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95, window_size= 3)
                sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
                sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
                
                td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                          fs=64, 
                                          label=int(label_value), 
                                          calc_sq=True)
                
                # Store if successful
                if td_stats and len(td_stats) > 0:
                    all_td_features.append(td_stats)
                    all_labels.append(int(label_value))
                    all_subject_ids.append(subject_id)
                    
            except Exception as e:
                print(f"   Error processing window: {e}")
                continue
    
    if not all_td_features:
        print(f"   No features extracted for S{subject_id}")
        return None
    
    print(f"   Extracted {len(all_td_features)} windows")
    
    # Create feature dataframe
    if isinstance(all_td_features[0], dict):
        td_df_temp = pd.DataFrame(all_td_features)
        td_array = td_df_temp.values
        td_columns = list(td_df_temp.columns)
    else:
        td_array = np.array(all_td_features)
        td_columns = [f'td_feature_{i}' for i in range(td_array.shape[1])]
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    td_scaler = StandardScaler()
    td_standardized = td_scaler.fit_transform(td_array)
    
    # Create final dataframe
    td_df = pd.DataFrame(td_standardized, columns=td_columns)
    
    # Combine with metadata
    result_df = pd.concat([
        pd.DataFrame({'subject_id': all_subject_ids, 'label': all_labels}),
        td_df
    ], axis=1)
    
    return result_df

def run_td_only_extraction_with_timing():
    """Main function for time-domain only feature extraction with timing"""
    
    # Configuration
    all_subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    output_file = '../../data/all_subjects_WESAD_td_features_timing.csv'
    
    print("="*70)
    print("TIME-DOMAIN ONLY FEATURE EXTRACTION WITH TIMING")
    print("="*70)
    
    # ===== TIMING STARTS HERE =====
    total_start_time = time.perf_counter()
    
    # Extract features for all subjects
    print(f"Extracting time-domain features for {len(all_subjects)} subjects")
    extraction_start_time = time.perf_counter()
    
    all_dataframes = []
    successful_subjects = []
    failed_subjects = []
    
    for subject_id in all_subjects:
        subject_start_time = time.perf_counter()
        
        try:
            subject_df = extract_td_features_for_subject(subject_id)
            if subject_df is not None:
                all_dataframes.append(subject_df)
                successful_subjects.append(subject_id)
                subject_time = time.perf_counter() - subject_start_time
                print(f"   S{subject_id}: {len(subject_df)} windows extracted in {subject_time:.2f}s")
            else:
                failed_subjects.append(subject_id)
                print(f"   S{subject_id}: FAILED")
        except Exception as e:
            failed_subjects.append(subject_id)
            print(f"   S{subject_id}: ERROR - {e}")
    
    extraction_time = time.perf_counter() - extraction_start_time
    print(f"\nFeature extraction completed in {extraction_time:.2f} seconds")
    
    # Combine all dataframes
    if all_dataframes:
        print(f"Combining {len(all_dataframes)} subject dataframes...")
        combine_start = time.perf_counter()
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
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
    print(f"Feature Extraction Time:  {extraction_time:.3f} seconds")
    print(f"Data Combination Time:    {combine_time:.3f} seconds")
    print(f"TOTAL TIME:              {total_time:.3f} seconds")
    
    print(f"\nAverage time per subject: {extraction_time/len(all_subjects):.3f} seconds")
    
    print("\nExtraction Summary:")
    print(f"Successful subjects: {len(successful_subjects)} - {successful_subjects}")
    if failed_subjects:
        print(f"Failed subjects: {len(failed_subjects)} - {failed_subjects}")
    
    print(f"\nFinal dataset:")
    print(f"Shape: {combined_df.shape}")
    print(f"Output file: {output_file}")
    
    # Show feature types
    feature_cols = [col for col in combined_df.columns if col not in ['subject_id', 'label']]
    print(f"Time-domain features: {len(feature_cols)}")

    # Remove duplicate columns if any
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    if len(feature_cols) > 0:
        print(f"Feature names: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    
    if 'label' in combined_df.columns:
        label_dist = combined_df['label'].value_counts().sort_index()
        print(f"Label distribution: {dict(label_dist)}")
    
    if 'subject_id' in combined_df.columns:
        subject_dist = combined_df['subject_id'].value_counts().sort_index()
        print(f"Subject distribution: {dict(subject_dist)}")
    
    # Return timing information for comparison
    return {
        'total_time': total_time,
        'extraction_time': extraction_time,
        'combine_time': combine_time,
        'avg_time_per_subject': extraction_time / len(all_subjects),
        'successful_subjects': len(successful_subjects),
        'total_subjects': len(all_subjects),
        'final_dataset_shape': combined_df.shape,
        'num_td_features': len(feature_cols)
    }

# Run the TD-only pipeline
if __name__ == "__main__":
    print("Starting time-domain only feature extraction...")
    results = run_td_only_extraction_with_timing()

    #save results to csv
    results_df = pd.DataFrame([results])
    results_df.to_csv('WESAD_td_extraction_results.csv', index=False)

    if results:
        print(f"\nTD-only pipeline completed successfully!")
        print(f"Total processing time: {results['total_time']:.2f} seconds")
        
    else:
        print(f"\nTD-only pipeline failed!")

#%%