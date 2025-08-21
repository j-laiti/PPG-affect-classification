# Debug script to test TD feature extraction on a specific Wellby session
#%% This will help identify where the feature extraction is failing

import numpy as np
import pandas as pd
import sys
import os

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize
from preprocessing.peak_detection import threshold_peakdetection

print("\n1. Loading raw PPG data...")
ppg_data = pd.read_csv("../../data/Wellby/selected_PPG_data.csv")

raw_signal = ppg_data['3871CA49-BC2A-4AD6-B70D-3A216EB16B61'].dropna().values

#%% Configuration
TARGET_SESSION = '3871CA49-BC2A-4AD6-B70D-3A216EB16B61'
FS = 50  # Sampling frequency for Wellby

def debug_session_extraction(session_id, verbose=True):
    """
    Debug TD feature extraction for a specific session
    """
    print(f"="*70)
    print(f"DEBUGGING TD FEATURE EXTRACTION FOR SESSION: {session_id}")
    print(f"="*70)
    
    try:
        # Step 1: Load raw PPG data
        print("\n1. Loading raw PPG data...")
        ppg_data = pd.read_csv("../../data/Wellby/selected_PPG_data.csv")
        
        if session_id not in ppg_data.columns:
            print(f"ERROR: Session {session_id} not found in PPG data!")
            print(f"Available sessions: {list(ppg_data.columns)}")
            return None
        
        raw_signal = ppg_data[session_id].dropna().values
        print(f"✓ Raw signal loaded successfully")
        print(f"  Length: {len(raw_signal)} samples")
        print(f"  Duration: ~{len(raw_signal)/FS:.1f} seconds")
        print(f"  Range: [{raw_signal.min():.3f}, {raw_signal.max():.3f}]")
        print(f"  Mean: {raw_signal.mean():.3f}, Std: {raw_signal.std():.3f}")
        
        if verbose:
            print(f"  First 10 values: {raw_signal[:10]}")
            print(f"  Last 10 values: {raw_signal[-10:]}")
        
        # Step 2: Check for invalid values
        print("\n2. Checking for invalid values...")
        nan_count = np.isnan(raw_signal).sum()
        inf_count = np.isinf(raw_signal).sum()
        zero_count = (raw_signal == 0).sum()
        
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        print(f"  Zero values: {zero_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  WARNING: Invalid values detected!")
        
        # Step 3: Clean signal (remove NaN)
        print("\n3. Cleaning signal...")
        clean_signal = raw_signal[~np.isnan(raw_signal)]
        
        if len(clean_signal) == 0:
            print("ERROR: No valid values after cleaning!")
            return None
            
        print(f"✓ Clean signal length: {len(clean_signal)} samples")
        print(f"  Removed: {len(raw_signal) - len(clean_signal)} invalid values")
        
        # Step 4: Standardize signal
        print("\n4. Standardizing signal...")
        try:
            standardized_signal = standardize(clean_signal)
            print(f"✓ Signal standardized successfully")
            print(f"  New range: [{standardized_signal.min():.3f}, {standardized_signal.max():.3f}]")
            print(f"  New mean: {standardized_signal.mean():.6f}, Std: {standardized_signal.std():.3f}")
        except Exception as e:
            print(f"ERROR during standardization: {e}")
            return None
        
        # Step 5: Apply bandpass filter
        print("\n5. Applying bandpass filter (0.5-10 Hz)...")
        try:
            bp_signal = bandpass_filter(standardized_signal, 0.5, 10, FS, order=2)
            print(f"✓ Bandpass filter applied successfully")
            print(f"  Filtered signal length: {len(bp_signal)}")
            print(f"  Range: [{bp_signal.min():.3f}, {bp_signal.max():.3f}]")
            print(f"  Mean: {bp_signal.mean():.6f}, Std: {bp_signal.std():.3f}")
            
            # Check for issues
            if np.isnan(bp_signal).sum() > 0:
                print(f"  ⚠️  WARNING: {np.isnan(bp_signal).sum()} NaN values after filtering!")
            if np.isinf(bp_signal).sum() > 0:
                print(f"  ⚠️  WARNING: {np.isinf(bp_signal).sum()} Inf values after filtering!")
                
        except Exception as e:
            print(f"ERROR during bandpass filtering: {e}")
            return None
        
        # Step 6: Apply moving average
        print("\n6. Applying moving average filter (window=5)...")
        try:
            smoothed_signal = moving_average_filter(bp_signal, window_size=5)
            print(f"✓ Moving average applied successfully")
            print(f"  Smoothed signal length: {len(smoothed_signal)}")
            print(f"  Range: [{smoothed_signal.min():.3f}, {smoothed_signal.max():.3f}]")
            print(f"  Mean: {smoothed_signal.mean():.6f}, Std: {smoothed_signal.std():.3f}")
            
            # Check for issues
            if np.isnan(smoothed_signal).sum() > 0:
                print(f"  ⚠️  WARNING: {np.isnan(smoothed_signal).sum()} NaN values after smoothing!")
            if np.isinf(smoothed_signal).sum() > 0:
                print(f"  ⚠️  WARNING: {np.isinf(smoothed_signal).sum()} Inf values after smoothing!")
                
        except Exception as e:
            print(f"ERROR during moving average: {e}")
            return None
        
        # Step 7: Test peak detection manually
        print("\n7. Testing peak detection...")
        try:
            peaks = threshold_peakdetection(smoothed_signal, FS)
            print(f"✓ Peak detection completed")
            print(f"  Peaks found: {len(peaks)} peaks")
            print(f"  Peak rate: {len(peaks)/(len(smoothed_signal)/FS)*60:.1f} BPM")
            
            if len(peaks) < 5:
                print(f"  ⚠️  WARNING: Very few peaks detected! This might cause feature extraction to fail.")
            
            if verbose and len(peaks) > 0:
                print(f"  First 5 peak indices: {peaks[:5]}")
                
        except Exception as e:
            print(f"ERROR during peak detection: {e}")
            return None
        
        # Step 8: Test get_ppg_features
        print("\n8. Testing get_ppg_features()...")
        try:
            # Test with label=0 (doesn't matter for debugging)
            td_features = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                         fs=FS, 
                                         label=0, 
                                         calc_sq=False)
            
            if td_features is None:
                print("ERROR: get_ppg_features() returned None!")
                print("  This suggests the function failed internally")
                return None
            else:
                print(f"✓ get_ppg_features() completed successfully")
                
                if isinstance(td_features, dict):
                    print(f"  Returned dict with {len(td_features)} features:")
                    for key, value in td_features.items():
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.3f}")
                        else:
                            print(f"    {key}: {value}")
                elif isinstance(td_features, list):
                    print(f"  Returned list with {len(td_features)} features:")
                    print(f"    Values: {td_features}")
                else:
                    print(f"  Returned {type(td_features)}: {td_features}")
                    
        except Exception as e:
            print(f"ERROR during get_ppg_features(): {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            return None
        
        print(f"\n{'='*70}")
        print("DEBUGGING COMPLETED SUCCESSFULLY!")
        print("✓ All steps passed - TD feature extraction should work for this session")
        print(f"{'='*70}")
        
        return td_features
        
    except Exception as e:
        print(f"\nFATAL ERROR during debugging: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_working_session():
    """
    Compare the problematic session with a known working session
    """
    print(f"\n{'='*70}")
    print("COMPARING WITH OTHER SESSIONS")
    print(f"{'='*70}")
    
    # Load session info to find other sessions
    session_info = pd.read_csv("../../data/Wellby/combined_sim_features.csv")
    other_sessions = session_info['Session_ID'].tolist()
    
    # Remove the problematic session
    if TARGET_SESSION in other_sessions:
        other_sessions.remove(TARGET_SESSION)
    
    print(f"Testing a few other sessions for comparison...")
    
    for i, session in enumerate(other_sessions[:3]):  # Test first 3
        print(f"\n--- Testing session {i+1}: {session} ---")
        result = debug_session_extraction(session, verbose=False)
        if result is not None:
            print(f"✓ Session {session} works fine")
        else:
            print(f"✗ Session {session} also fails!")

# Main execution
if __name__ == "__main__":
    print("Starting TD feature extraction debugging...")
    
    # Debug the problematic session
    result = debug_session_extraction(TARGET_SESSION)
    
    # Compare with other sessions if requested
    print(f"\nWould you like to compare with other sessions? (This helps identify if it's a systematic issue)")
    # Uncomment the line below to automatically run comparison
    # compare_with_working_session()
    
    if result is not None:
        print(f"\n🎉 SUCCESS: TD features extracted successfully!")
    else:
        print(f"\n❌ FAILURE: TD feature extraction failed for session {TARGET_SESSION}")
        print(f"\nNext steps:")
        print(f"1. Check the error messages above to identify the failing step")
        print(f"2. Look at the signal characteristics - might be too short/noisy")
        print(f"3. Consider adding more robust error handling to the main extraction")
# %%
