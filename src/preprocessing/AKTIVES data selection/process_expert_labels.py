"""
AKTIVES Expert Label Processing

Processes raw expert stress annotations and generates majority vote labels.
Creates ExpertLabels_with_majority.csv files that serve as input for window generation.

Expert annotations are in columns 3, 5, and 7 (0-indexed) of the raw ExpertLabels.csv.
Majority vote requires ≥2 experts to agree on "Stress".

Usage:
    python process_expert_labels.py
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import pandas as pd
from pathlib import Path

#%% Expert Label Processing Functions

def process_expert_labels(file_path):
    """
    Process a single ExpertLabels.csv file and add majority vote column.
    
    Expert stress labels are at positions:
    - Column 3: Expert 1 Stress/No Stress
    - Column 5: Expert 2 Stress/No Stress  
    - Column 7: Expert 3 Stress/No Stress
    
    Args:
        file_path: Path to ExpertLabels.csv file
        
    Returns:
        Tuple of (processed_dataframe, success_boolean)
    """
    try:
        df = pd.read_csv(file_path)
        
        print(f"Columns in file: {list(df.columns)}")
        print(f"First few rows:\n{df.head(3)}")
        
        # Expert stress label column positions (0-indexed)
        stress_col_indices = [3, 5, 7]
        stress_cols = [df.columns[i] for i in stress_col_indices if i < len(df.columns)]
        
        print(f"Using stress columns at positions {stress_col_indices}: {stress_cols}")
        
        # Create binary stress indicators (1 for Stress, 0 for No Stress)
        stress_binary = pd.DataFrame()
        for i, col in enumerate(stress_cols):
            col_name = f'Expert{i+1}_Stress_Binary'
            stress_binary[col_name] = (df[col] == 'Stress').astype(int)
            print(f"Expert {i+1} stress count: {stress_binary[col_name].sum()}")
        
        # Calculate majority vote (≥2 experts agree on stress)
        df['Stress_Vote_Count'] = stress_binary.sum(axis=1)
        df['Majority_Stress_Vote'] = (df['Stress_Vote_Count'] >= 2).astype(int)
        df['Majority_Stress_Label'] = df['Majority_Stress_Vote'].map({
            1: 'Stress', 
            0: 'No Stress'
        })
        
        print(f"Vote count distribution:\n{df['Stress_Vote_Count'].value_counts().sort_index()}")
        print(f"Majority stress intervals: {df['Majority_Stress_Vote'].sum()}")
        
        return df, True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def process_all_participants(base_folder):
    """
    Process all participants' expert label files in a cohort.
    
    Args:
        base_folder: Path to cohort folder (e.g., "Aktives/Typically Developed")
        
    Returns:
        Dictionary with processing results for each participant/game
    """
    base_path = Path(base_folder)
    results = {}
    
    # Iterate through participant folders (C16-C25)
    for participant_folder in base_path.iterdir():
        if participant_folder.is_dir() and participant_folder.name.startswith('C'):
            participant_id = participant_folder.name
            results[participant_id] = {}
            
            print(f"\n{'='*60}")
            print(f"Processing participant: {participant_id}")
            print(f"{'='*60}")
            
            # Process both games
            for game in ['CatchAPet', 'LeapBall']:
                game_path = participant_folder / game
                
                if game_path.exists():
                    expert_labels_path = game_path / 'ExpertLabels.csv'
                    
                    if expert_labels_path.exists():
                        print(f"\n--- Processing {game} ---")
                        
                        # Process the file
                        processed_df, success = process_expert_labels(expert_labels_path)
                        
                        if success:
                            # Save processed file
                            output_path = game_path / 'ExpertLabels_with_majority.csv'
                            processed_df.to_csv(output_path, index=False)
                            
                            # Store results
                            results[participant_id][game] = {
                                'original_path': expert_labels_path,
                                'output_path': output_path,
                                'total_intervals': len(processed_df),
                                'stress_intervals': processed_df['Majority_Stress_Vote'].sum(),
                                'no_stress_intervals': (processed_df['Majority_Stress_Vote'] == 0).sum()
                            }
                            
                            print(f"SUCCESS: Processed {len(processed_df)} intervals")
                            print(f"  Stress intervals: {processed_df['Majority_Stress_Vote'].sum()}")
                            print(f"  No Stress intervals: {(processed_df['Majority_Stress_Vote'] == 0).sum()}")
                            
                            # Show sample
                            print("\nSample processed rows:")
                            sample_cols = ['Stress_Vote_Count', 'Majority_Stress_Vote', 'Majority_Stress_Label']
                            print(processed_df[sample_cols].head())
                            
                        else:
                            print(f"FAILED: Could not process {expert_labels_path}")
                    else:
                        print(f"  No ExpertLabels.csv found in {game}")
                else:
                    print(f"  No {game} folder found")
    
    return results

#%% Main Execution

if __name__ == "__main__":
    # Process all cohorts
    cohorts = [
        "../../data/Aktives/Typically Developed",
        "../../data/Aktives/Dyslexia",
        "../../data/Aktives/Intellectual Disabilities",
        "../../data/Aktives/Obstetric Brachial Plexus Injuries"
    ]
    
    for cohort_path in cohorts:
        print(f"\n{'='*80}")
        print(f"Processing cohort: {cohort_path}")
        print(f"{'='*80}")
        
        results = process_all_participants(cohort_path)
        
        # Summary
        print(f"\n{'='*60}")
        print("COHORT SUMMARY")
        print(f"{'='*60}")
        
        total_stress = 0
        total_intervals = 0
        
        for participant_id, games in results.items():
            for game, stats in games.items():
                if stats:
                    total_stress += stats['stress_intervals']
                    total_intervals += stats['total_intervals']
                    print(f"{participant_id} {game}: {stats['stress_intervals']}/{stats['total_intervals']} stress")
        
        if total_intervals > 0:
            print(f"\nOverall: {total_stress}/{total_intervals} stress intervals "
                  f"({total_stress/total_intervals*100:.1f}%)")