"""
PPG Feature Extraction Module

Extracts time-domain HRV features from PPG signals including peak detection,
RR interval calculation, and signal quality assessment. Adapted from Heo et al. (2021).

Reference: https://doi.org/10.1109/ACCESS.2021.3060441

Author: Justin Laiti (adapted from Heo et al., 2021)
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import numpy as np
from .peak_detection import threshold_peakdetection

#%% RR Interval Calculation

def calc_RRI(peaklist, fs):
    """
    Calculate RR intervals from detected peaks with outlier removal.
    
    Args:
        peaklist: List of peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        RR_list_e: Filtered RR intervals in ms
        RR_diff: Absolute differences between consecutive RR intervals
        RR_sqdiff: Squared differences between consecutive RR intervals
    """
    if len(peaklist) < 2:
        return [], [], []

    # Convert peak intervals to milliseconds
    RR_list = [
        (peaklist[i + 1] - peaklist[i]) / fs * 1000.0 
        for i in range(len(peaklist) - 1)
    ]
    
    # Remove RR intervals outside physiological range (300-2000 ms)
    RR_list = [rr for rr in RR_list if 300 <= rr <= 2000]

    if not RR_list:
        return [], [], []

    # Remove statistical outliers (>1.5 SD from mean)
    mean_RR = np.mean(RR_list)
    std_RR = np.std(RR_list)
    lower_bound = mean_RR - (1.5 * std_RR)
    upper_bound = mean_RR + (1.5 * std_RR)

    RR_list_e = [rr for rr in RR_list if lower_bound <= rr <= upper_bound]

    # Compute differences for time-domain features
    RR_diff = np.abs(np.diff(RR_list_e))
    RR_sqdiff = np.diff(RR_list_e) ** 2

    return RR_list_e, RR_diff, RR_sqdiff


def calc_heartrate(RR_list):
    """
    Calculate instantaneous heart rate from RR intervals.
    
    Args:
        RR_list: List of RR intervals in ms
        
    Returns:
        HR: List of heart rate values in BPM
    """
    HR = []
    window_size = 10

    for val in RR_list:
        if 400 < val < 1500:
            heart_rate = 60000.0 / val
        elif (0 < val < 400) or val > 1500:
            # Use recent average for outliers
            heart_rate = np.mean(HR[-window_size:]) if HR else 60.0
        else:
            heart_rate = 0.0
        HR.append(heart_rate)

    return HR

#%% Time-Domain HRV Features

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff):
    """
    Calculate standard time-domain HRV features.
    
    Args:
        RR_list: Filtered RR intervals in ms
        RR_diff: Absolute differences between consecutive RR intervals
        RR_sqdiff: Squared differences between consecutive RR intervals
        
    Returns:
        features: Dictionary of time-domain HRV metrics
    """
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    
    # NN interval statistics
    meanNN = np.mean(RR_list)
    SDNN = np.std(RR_list)
    medianNN = np.median(np.abs(RR_list))
    
    # Successive difference statistics
    meanSD = np.mean(RR_diff)
    SDSD = np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    # pNN metrics (percentage of intervals differing by >X ms)
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / len(RR_diff) * 100
    pNN50 = len(NN50) / len(RR_diff) * 100

    features = {
        'HR_mean': HR_mean,
        'HR_std': HR_std,
        'meanNN': meanNN,
        'SDNN': SDNN,
        'medianNN': medianNN,
        'meanSD': meanSD,
        'SDSD': SDSD,
        'RMSSD': RMSSD,
        'pNN20': pNN20,
        'pNN50': pNN50
    }

    return features

#%% Signal Quality Assessment

def calculate_signal_quality(raw_ppg_signal, peaklist, fs=50):
    """
    Estimate PPG signal quality based on SNR and RR interval plausibility.
    
    Args:
        raw_ppg_signal: Raw PPG signal array
        peaklist: List of detected peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        sqi: Signal quality index (0-1, higher is better)
    """
    # Calculate absolute SNR
    snr = (
        np.abs(np.std(raw_ppg_signal) / np.mean(raw_ppg_signal)) 
        if np.mean(raw_ppg_signal) != 0 else 0
    )

    # Calculate RR interval statistics
    if len(peaklist) < 2:
        rr_mean, rr_std = 0, 0
    else:
        rr_intervals = [
            (peaklist[i + 1] - peaklist[i]) / fs * 1000 
            for i in range(len(peaklist) - 1)
        ]
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)

    # Compute SQI components
    rr_std_clipped = min(rr_std, 1000)
    rr_mean_penalty = 1 if rr_mean < 300 or rr_mean > 2000 else 0

    # Combine into final SQI (weighted average)
    sqi = max(0, 1 - (
        0.4 * (1 - min(snr / 100, 1)) + 
        0.4 * (rr_std_clipped / 1000) + 
        0.2 * rr_mean_penalty
    ))

    return sqi

#%% Main Feature Extraction

def get_ppg_features(ppg_seg, fs, label, raw_ppg_signal=0, calc_sq=False):
    """
    Extract complete feature set from PPG segment.
    
    Args:
        ppg_seg: Filtered PPG signal segment
        fs: Sampling frequency in Hz
        label: Ground truth label for this segment
        raw_ppg_signal: Raw unfiltered PPG signal (for SQI calculation)
        calc_sq: Whether to calculate signal quality index
        
    Returns:
        total_features: Dictionary of all extracted features + label
    """
    # Require minimum 10 seconds of data
    if len(ppg_seg) < fs * 10:
        print(f"Less than 10 sec of data")
        return {}

    # Detect peaks
    peak = threshold_peakdetection(ppg_seg, fs)

    # Calculate RR intervals
    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    print(f"RR intervals before filtering: {RR_list}")
    
    if len(RR_list) <= 3:
        return []
    
    # Extract time-domain features
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff)
    
    total_features = {**td_features}
    total_features["label"] = label

    # Add signal quality if requested
    if calc_sq:
        sqi = calculate_signal_quality(raw_ppg_signal, peak, fs=fs)
        total_features["sqi"] = sqi

    return total_features