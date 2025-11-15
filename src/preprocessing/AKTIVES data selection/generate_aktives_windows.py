"""
AKTIVES Analysis Window Generation

Generates analysis windows from AKTIVES expert stress labels.
Creates overlapping time windows for stress (20s) and non-stress (30s) segments
based on expert annotations.

Usage:
    python generate_aktives_windows.py
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import pandas as pd
from pathlib import Path

#%% Window Generation Functions

def find_stress_nonstress_intervals(df):
    """
    Identify consecutive segments of stress/non-stress labels.
    
    Args:
        df: DataFrame with 'Majority_Stress_Vote' column
        
    Returns:
        List of segment dictionaries with start/end indices and duration
    """
    stress_sequence = df['Majority_Stress_Vote'].values
    n_intervals = len(stress_sequence)
    
    segments = []
    current_label = None
    segment_start = 0
    
    for i, label in enumerate(stress_sequence):
        if label != current_label:
            if current_label is not None:
                segments.append({
                    'label': 'stress' if current_label == 1 else 'non-stress',
                    'start_idx': segment_start,
                    'end_idx': i - 1,
                    'length_intervals': i - segment_start,
                    'length_seconds': (i - segment_start) * 10
                })
            segment_start = i
            current_label = label
    
    # Add final segment
    if current_label is not None:
        segments.append({
            'label': 'stress' if current_label == 1 else 'non-stress',
            'start_idx': segment_start,
            'end_idx': n_intervals - 1,
            'length_intervals': n_intervals - segment_start,
            'length_seconds': (n_intervals - segment_start) * 10
        })
    
    return segments


def extract_analysis_intervals_overlapping(df, segments):
    """
    Extract overlapping analysis windows from stress/non-stress segments.
    
    Stress windows: 20s (2 intervals) expanded ±10s, with 10s overlap
    Non-stress windows: 30s (3 intervals) with 10s overlap
    
    Args:
        df: Original DataFrame
        segments: List of segment dictionaries from find_stress_nonstress_intervals
        
    Returns:
        Dictionary with 'stress' and 'non-stress' window lists
    """
    analysis_intervals = {
        'stress': [],
        'non-stress': []
    }
    
    n_intervals = len(df)
    
    # Process stress segments (minimum 2 intervals = 20s)
    for segment in segments:
        if segment['label'] == 'stress' and segment['length_intervals'] >= 2:
            segment_start = segment['start_idx']
            segment_end = segment['end_idx']
            
            window_start = segment_start
            window_count = 0
            
            # Create overlapping 20s windows
            while window_start + 1 <= segment_end:
                window_end = window_start + 1
                
                # Expand by ±10s when possible
                expanded_start = max(0, window_start - 1)
                expanded_end = min(n_intervals - 1, window_end + 1)
                
                # Handle edge cases
                if window_start == 0:
                    expanded_end = min(n_intervals - 1, window_end + 2)
                elif window_end == n_intervals - 1:
                    expanded_start = max(0, window_start - 2)
                
                analysis_intervals['stress'].append({
                    'core_start_idx': window_start,
                    'core_end_idx': window_end,
                    'expanded_start_idx': expanded_start,
                    'expanded_end_idx': expanded_end,
                    'window_id': window_count,
                    'segment_id': f"stress_{segment_start}_{segment_end}"
                })
                
                window_count += 1
                window_start += 1
    
    # Process non-stress segments (minimum 3 intervals = 30s)
    for segment in segments:
        if segment['label'] == 'non-stress' and segment['length_intervals'] >= 3:
            segment_start = segment['start_idx']
            segment_end = segment['end_idx']
            
            window_start = segment_start
            window_count = 0
            
            # Create overlapping 30s windows
            while window_start + 2 <= segment_end:
                window_end = window_start + 2
                
                analysis_intervals['non-stress'].append({
                    'start_idx': window_start,
                    'end_idx': window_end,
                    'window_id': window_count,
                    'segment_id': f"nonstress_{segment_start}_{segment_end}"
                })
                
                window_count += 1
                window_start += 3  # 10s overlap (30s - 20s)
    
    return analysis_intervals


def generate_analysis_windows_table(df, participant_id):
    """
    Generate analysis windows table with time intervals and labels.
    
    Args:
        df: DataFrame with expert labels
        participant_id: Identifier for this participant/game combination
        
    Returns:
        List of analysis window records
    """
    segments = find_stress_nonstress_intervals(df)
    windows = extract_analysis_intervals_overlapping(df, segments)
    
    analysis_records = []
    
    # Process stress windows
    for window in windows['stress']:
        expanded_start_idx = window['expanded_start_idx']
        expanded_end_idx = window['expanded_end_idx']
        
        # Calculate time intervals
        start_time = 0 if expanded_start_idx == 0 else (expanded_start_idx * 10 + 5)
        end_time = expanded_end_idx * 10 + 5
        
        analysis_records.append({
            'Participant': participant_id,
            'Interval_Start': start_time,
            'Interval_End': end_time,
            'Label': 1,
            'Window_Type': 'stress',
            'Core_Start': window['core_start_idx'] * 10,
            'Core_End': (window['core_end_idx'] + 1) * 10,
            'Window_ID': window['window_id'],
            'Segment_ID': window['segment_id']
        })
    
    # Process non-stress windows
    for window in windows['non-stress']:
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        
        start_time = 0 if start_idx == 0 else start_idx * 10
        end_time = (end_idx + 1) * 10
        
        analysis_records.append({
            'Participant': participant_id,
            'Interval_Start': start_time,
            'Interval_End': end_time,
            'Label': 0,
            'Window_Type': 'non-stress',
            'Core_Start': start_time,
            'Core_End': end_time,
            'Window_ID': window['window_id'],
            'Segment_ID': window['segment_id']
        })
    
    return analysis_records

#%% Processing Functions

def process_all_participants_to_table(base_folder, output_file="analysis_windows.csv"):
    """
    Process all participants and create the analysis windows table.
    
    Args:
        base_folder: Path to cohort folder containing participant data
        output_file: Output CSV filename
        
    Returns:
        Tuple of (simplified_table, detailed_table)
    """
    base_path = Path(base_folder)
    all_records = []
    
    # Process each participant
    for participant_folder in base_path.iterdir():
        if participant_folder.is_dir() and participant_folder.name.startswith('C'):
            participant_id = participant_folder.name
            
            print(f"Processing {participant_id}...")
            
            # Process both games
            for game in ['CatchAPet', 'LeapBall']:
                labels_file = participant_folder / game / 'ExpertLabels_with_majority.csv'
                
                if labels_file.exists():
                    try:
                        df = pd.read_csv(labels_file)
                        game_records = generate_analysis_windows_table(
                            df, f"{participant_id}_{game}"
                        )
                        all_records.extend(game_records)
                        
                        stress_count = len([r for r in game_records if r['Label'] == 1])
                        nonstress_count = len([r for r in game_records if r['Label'] == 0])
                        print(f"  {game}: {stress_count} stress windows, "
                              f"{nonstress_count} non-stress windows")
                        
                    except Exception as e:
                        print(f"  {game}: Error - {e}")
    
    # Create DataFrames
    df_windows = pd.DataFrame(all_records)
    simplified_table = df_windows[['Participant', 'Interval_Start', 'Interval_End', 'Label']].copy()
    simplified_table = simplified_table.sort_values(['Participant', 'Interval_Start'])
    
    # Save both versions
    simplified_table.to_csv(output_file, index=False)
    df_windows.to_csv(output_file.replace('.csv', '_detailed.csv'), index=False)
    
    print(f"\nSaved simplified table to: {output_file}")
    print(f"Saved detailed table to: {output_file.replace('.csv', '_detailed.csv')}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Participants: {df_windows['Participant'].nunique()}")
    print(f"  Stress windows: {len(df_windows[df_windows['Label'] == 1])}")
    print(f"  Non-stress windows: {len(df_windows[df_windows['Label'] == 0])}")
    print(f"  Total windows: {len(df_windows)}")
    
    return simplified_table, df_windows

#%% Main Execution

if __name__ == "__main__":
    # Process all cohorts
    cohorts = {
        'TD': '../../data/Aktives/Typically Developed',
        'dyslexia': '../../data/Aktives/Dyslexia',
        'ID': '../../data/Aktives/Intellectual Disabilities',
        'OBPI': '../../data/Aktives/Obstetric Brachial Plexus Injuries'
    }
    
    for cohort_name, cohort_path in cohorts.items():
        print(f"\n{'='*60}")
        print(f"Processing {cohort_name} cohort...")
        print(f"{'='*60}")
        
        output_file = f"../../data/Aktives/analysis_windows/analysis_windows_{cohort_name}.csv"
        
        try:
            simplified, detailed = process_all_participants_to_table(
                cohort_path, 
                output_file
            )
        except Exception as e:
            print(f"Error processing {cohort_name}: {e}")