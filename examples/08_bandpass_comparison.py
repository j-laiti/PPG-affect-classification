import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_hrv_data(folder_path):
    """
    Load HRV data from all subject CSV files within a given directory
    
    Args:
        folder_path (str): Path to the folder containing subject CSV files
    
    Returns:
        pandas.DataFrame: Combined dataframe with all subjects' HRV data
    """
    all_data = []
    folder_path = Path(folder_path)
    
    # Get all CSV files that start with 's' followed by numbers (e.g., s2.csv, s3.csv)
    csv_files = [f for f in folder_path.glob('S*.csv')]
    
    print(f"Found {len(csv_files)} subject files in {folder_path}")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract subject ID from filename (e.g., 's2.csv' -> 's2')
            subject_id = csv_file.stem
            df['subject_id'] = subject_id
            df['file_name'] = csv_file.name
            all_data.append(df)
            print(f"Loaded {csv_file.name}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total rows from {len(all_data)} subjects")
        return combined_df
    else:
        print(f"No data found in {folder_path}")
        return pd.DataFrame()

def identify_hrv_columns(df):
    """
    Identify HRV feature columns (exclude index, subject_id, stress, file_name columns)
    
    Args:
        df (pandas.DataFrame): DataFrame with HRV features
    
    Returns:
        list: List of HRV feature column names
    """
    exclude_cols = ['index', 'subject_id', 'stress', 'file_name', 'Unnamed: 0']
    hrv_cols = [col for col in df.columns if col not in exclude_cols]
    return hrv_cols

def calculate_subject_means(df, hrv_columns):
    """
    Calculate mean HRV values for each subject
    
    Args:
        df (pandas.DataFrame): DataFrame with HRV data
        hrv_columns (list): List of HRV feature column names
    
    Returns:
        pandas.DataFrame: DataFrame with mean HRV values per subject
    """
    # Group by subject and calculate means for HRV columns
    subject_means = df.groupby('subject_id')[hrv_columns].mean().reset_index()
    return subject_means

def compare_filter_results(folder1, folder2, folder3, filter_names=None):
    """
    Compare HRV results across three different filter configurations
    
    Args:
        folder1, folder2, folder3 (str): Paths to the three filter result folders
        filter_names (list): Names for the three filters (optional)
    
    Returns:
        dict: Dictionary containing comparison results
    """
    if filter_names is None:
        filter_names = ['Filter_1', 'Filter_2', 'Filter_3']
    
    # Load data from all three folders
    print("Loading data from three filter configurations...")
    df1 = load_hrv_data(folder1)
    df2 = load_hrv_data(folder2)
    df3 = load_hrv_data(folder3)
    
    if df1.empty or df2.empty or df3.empty:
        print("Error: One or more datasets is empty")
        return None
    
    # Identify HRV columns (should be the same across all datasets)
    hrv_cols = identify_hrv_columns(df1)
    print(f"HRV columns identified: {hrv_cols}")
    
    # Calculate subject means for each filter
    means1 = calculate_subject_means(df1, hrv_cols)
    means2 = calculate_subject_means(df2, hrv_cols)
    means3 = calculate_subject_means(df3, hrv_cols)
    
    # Merge the datasets on subject_id
    merged = means1.merge(means2, on='subject_id', suffixes=('_f1', '_f2'))
    merged = merged.merge(means3, on='subject_id')
    
    # Rename columns for filter 3
    for col in hrv_cols:
        if col in merged.columns:
            merged = merged.rename(columns={col: f"{col}_f3"})
    
    # Calculate percentage differences
    comparison_results = {}
    
    for hrv_feature in hrv_cols:
        col1 = f"{hrv_feature}_f1"
        col2 = f"{hrv_feature}_f2"
        col3 = f"{hrv_feature}_f3"
        
        if all(col in merged.columns for col in [col1, col2, col3]):
            # Calculate percentage differences (using filter 1 as reference)
            diff_1_2 = ((merged[col2] - merged[col1]) / merged[col1] * 100).abs()
            diff_1_3 = ((merged[col3] - merged[col1]) / merged[col1] * 100).abs()
            diff_2_3 = ((merged[col3] - merged[col2]) / merged[col2] * 100).abs()
            
            comparison_results[hrv_feature] = {
                'mean_diff_f1_f2': diff_1_2.mean(),
                'mean_diff_f1_f3': diff_1_3.mean(),
                'mean_diff_f2_f3': diff_2_3.mean(),
                'max_diff_f1_f2': diff_1_2.max(),
                'max_diff_f1_f3': diff_1_3.max(),
                'max_diff_f2_f3': diff_2_3.max(),
                'filter1_mean': merged[col1].mean(),
                'filter2_mean': merged[col2].mean(),
                'filter3_mean': merged[col3].mean()
            }
    
    return {
        'comparison_results': comparison_results,
        'merged_data': merged,
        'hrv_columns': hrv_cols,
        'filter_names': filter_names
    }

def summarize_comparison(results):
    """
    Create a summary of the comparison results
    
    Args:
        results (dict): Results from compare_filter_results
    
    Returns:
        pandas.DataFrame: Summary dataframe
    """
    if results is None:
        return None
    
    comparison_results = results['comparison_results']
    filter_names = results['filter_names']
    
    summary_data = []
    for feature, metrics in comparison_results.items():
        summary_data.append({
            'HRV_Feature': feature,
            f'Mean_Diff_{filter_names[0]}_vs_{filter_names[1]}_%': round(metrics['mean_diff_f1_f2'], 2),
            f'Mean_Diff_{filter_names[0]}_vs_{filter_names[2]}_%': round(metrics['mean_diff_f1_f3'], 2),
            f'Mean_Diff_{filter_names[1]}_vs_{filter_names[2]}_%': round(metrics['mean_diff_f2_f3'], 2),
            f'Max_Diff_{filter_names[0]}_vs_{filter_names[1]}_%': round(metrics['max_diff_f1_f2'], 2),
            f'Max_Diff_{filter_names[0]}_vs_{filter_names[2]}_%': round(metrics['max_diff_f1_f3'], 2),
            f'Max_Diff_{filter_names[1]}_vs_{filter_names[2]}_%': round(metrics['max_diff_f2_f3'], 2),
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def plot_comparison(results):
    """
    Create visualization of the filter comparison
    
    Args:
        results (dict): Results from compare_filter_results
    """
    if results is None:
        return
    
    comparison_results = results['comparison_results']
    filter_names = results['filter_names']
    
    # Prepare data for plotting
    features = list(comparison_results.keys())
    mean_diffs_1_2 = [comparison_results[f]['mean_diff_f1_f2'] for f in features]
    mean_diffs_1_3 = [comparison_results[f]['mean_diff_f1_f3'] for f in features]
    mean_diffs_2_3 = [comparison_results[f]['mean_diff_f2_f3'] for f in features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(features))
    width = 0.25
    
    ax.bar(x - width, mean_diffs_1_2, width, label=f'{filter_names[0]} vs {filter_names[1]}', alpha=0.8)
    ax.bar(x, mean_diffs_1_3, width, label=f'{filter_names[0]} vs {filter_names[2]}', alpha=0.8)
    ax.bar(x + width, mean_diffs_2_3, width, label=f'{filter_names[1]} vs {filter_names[2]}', alpha=0.8)
    
    ax.set_xlabel('HRV Features')
    ax.set_ylabel('Mean Percentage Difference (%)')
    ax.set_title('HRV Feature Differences Across Filter Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # MODIFY THESE PATHS TO MATCH YOUR FOLDER STRUCTURE
    folder_path_1 = "../data/WESAD/subject_extracted_features_bp_time120"  # e.g., "0.5-10Hz_results"
    folder_path_2 = "../data/WESAD/subject_bp8_extracted_features_bp_time120"  # e.g., "0.2-10Hz_results" 
    folder_path_3 = "../data/WESAD/subject_bp02_extracted_features_bp_time120"  # e.g., "0.5-8Hz_results"
    
    # Optional: Give meaningful names to your filters
    filter_names = ["0.5-10Hz", "0.2-10Hz", "0.5-8Hz"]  # Modify as needed
    
    # Run the comparison
    print("Starting HRV filter comparison analysis...")
    results = compare_filter_results(folder_path_1, folder_path_2, folder_path_3, filter_names)
    
    if results:
        # Generate summary
        summary = summarize_comparison(results)
        
        if summary is not None:
            print("\n" + "="*50)
            print("SUMMARY OF HRV FILTER COMPARISON")
            print("="*50)
            print(summary.to_string(index=False))
            
            # Calculate overall average differences
            mean_cols = [col for col in summary.columns if 'Mean_Diff' in col]
            overall_averages = summary[mean_cols].mean()
            
            print("\n" + "="*50)
            print("OVERALL AVERAGE DIFFERENCES")
            print("="*50)
            for col, avg in overall_averages.items():
                print(f"{col}: {avg:.2f}%")
            
            # Save results
            summary.to_csv("hrv_filter_comparison_summary.csv", index=False)
            print(f"\nSummary saved to: hrv_filter_comparison_summary.csv")
            
            # Create visualization
            plot_comparison(results)
        
        print("\nAnalysis complete!")
    else:
        print("Analysis failed - please check your folder paths and data structure.")