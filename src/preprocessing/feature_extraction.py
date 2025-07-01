# feature_extraction_JL.py
import numpy as np
import math
from scipy.interpolate import UnivariateSpline
import sys
import types
# import nolds
from nolds.measures import sampen, corr_dim
from scipy import stats
from preprocessing.peak_detection import threshold_peakdetection

def calc_RRI(peaklist, fs):
    if len(peaklist) < 2:
        return [], [], []

    RR_list = [(peaklist[i + 1] - peaklist[i]) / fs * 1000.0 for i in range(len(peaklist) - 1)]
    
    # Remove RR intervals outside physiological range (300–2000 ms)
    RR_list = [rr for rr in RR_list if 300 <= rr <= 2000]

    if not RR_list:  # Handle case where all intervals are filtered out
        return [], [], []

    mean_RR = np.mean(RR_list)
    std_RR = np.std(RR_list)
    lower_bound, upper_bound = mean_RR - (1.5 * std_RR), mean_RR + (1.5 * std_RR)

    # Filter out outliers
    RR_list_e = [rr for rr in RR_list if lower_bound <= rr <= upper_bound]

    # Compute differences and squared differences
    RR_diff = np.abs(np.diff(RR_list_e))
    RR_sqdiff = np.diff(RR_list_e) ** 2

    return RR_list_e, RR_diff, RR_sqdiff


def calc_heartrate(RR_list):
    HR = []
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                heart_rate = np.mean(HR[-window_size:])
            else:
                heart_rate = 60.0
        else:
            heart_rate = 0.0
        HR.append(heart_rate)

    return HR

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff):
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / len(RR_diff) * 100
    pNN50 = len(NN50) / len(RR_diff) * 100

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50}

    return features

def get_ppg_features(ppg_seg, fs, label, raw_ppg_signal=0, calc_sq=False):

    if len(ppg_seg) < fs * 10:  # Ensure at least 10 seconds of data
        print(f"Less than 10 sec of data")
        return {}

    peak = threshold_peakdetection(ppg_seg, fs)

    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    print(f"RR intervals before filtering: {RR_list}")
    if len(RR_list) <= 3:
        return []
    
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff)
    
    total_features = {**td_features}
    total_features["label"] = label

    # Calculate signal quality
    if calc_sq:
        sqi = calculate_signal_quality(raw_ppg_signal, peak, fs=fs)

        # Add SQI
        total_features["sqi"] = sqi

    return total_features


def calculate_signal_quality(raw_ppg_signal, peaklist, fs=50):
    """
    Estimate signal quality based on absolute SNR, RR interval plausibility, and rr_std.

    Parameters:
    - ppg_seg (array): Filtered PPG signal segment.
    - peaklist (list): Detected peaks in the signal.
    - fs (int): Sampling frequency of the PPG signal.

    Returns:
    - sqi (float): Signal quality index (0 to 1, higher is better).
    - quality_flags (dict): Detailed signal quality indicators.
    """
    # 1. Absolute SNR
    snr = np.abs(np.std(raw_ppg_signal) / np.mean(raw_ppg_signal)) if np.mean(raw_ppg_signal) != 0 else 0

    # 2. RR Intervals and Plausibility
    if len(peaklist) < 2:
        rr_mean, rr_std = 0, 0
    else:
        rr_intervals = [(peaklist[i + 1] - peaklist[i]) / fs * 1000 for i in range(len(peaklist) - 1)]
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)

    # 3. Combine into SQI
    rr_std_clipped = min(rr_std, 1000)  # Cap rr_std to avoid excessive influence
    rr_mean_penalty = 1 if rr_mean < 300 or rr_mean > 2000 else 0  # Penalty for implausible mean RR

    # SQI computation
    sqi = max(0, 1 - (0.4 * (1 - min(snr / 100, 1)) + 0.4 * (rr_std_clipped / 1000) + 0.2 * rr_mean_penalty))

    return sqi