"""
PPG Signal Preprocessing Filters

Preprocessing method based on this paper:
S. Heo, S. Kwon and J. Lee, "Stress Detection With Single PPG Sensor by Orchestrating Multiple Denoising and Peak-Detecting Methods," 
in IEEE Access, vol. 9, pp. 47777-47785, 2021, doi: 10.1109/ACCESS.2021.3060441.

Adapted March 3rd 2025 by Justin Laiti
Claude AI used to generate function docstrings.
"""""

import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply bandpass filter to PPG signal.
    
    Parameters:
    -----------
    data : array-like
        Input PPG signal
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int, default=2
        Filter order
        
    Returns:
    --------
    array
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y


def moving_average_filter(data, window_size=3):
    """
    Apply moving average filter to smooth PPG signal.
    
    Parameters:
    -----------
    data : array-like
        Input PPG signal
    window_size : int, default=3
        Size of moving average window
        
    Returns:
    --------
    array
        Smoothed signal
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def standardize(ppg_signal):
    """
    Standardize PPG signal (z-score normalization).
    
    Parameters:
    -----------
    ppg_signal : array-like
        Input PPG signal
        
    Returns:
    --------
    array
        Standardized signal
    """
    mean = ppg_signal.mean()
    std = ppg_signal.std()
    if std == 0:  # Prevent division by zero
        return np.zeros_like(ppg_signal)
    norm_signal = (ppg_signal - mean) / std
    return norm_signal


# === Simple Noise Elimination Methods ===

def simple_dynamic_threshold(clean_signal, fs, percentile, window_size = 3):
    """
    Calculate the dynamic threshold based on the 95th percentile of standard deviations
    across signal segments.

    Window size is in seconds and is converted to samples using the sampling frequency (fs).
    """
    step_size = window_size * fs
    segment_stds = [
        np.std(clean_signal[i:i + step_size])
        for i in range(0, len(clean_signal) - step_size, step_size)
    ]
    threshold = np.percentile(segment_stds, percentile)
    return segment_stds, threshold

def simple_noise_elimination(clean_signal, fs, threshold, window_size=3):
    """
    Eliminate noisy segments of the signal based on the calculated threshold.
    """
    clean_indices = []
    step_size = window_size * fs
    for i in range(0, len(clean_signal) - step_size, step_size):
        segment = clean_signal[i:i + step_size]
        if np.std(segment) < threshold:
            clean_indices.extend(range(i, i + step_size))
    # Return the cleaned signal and indices for WESAD
    return clean_signal[clean_indices], clean_indices


def simple_statistic_adaptive_threshold(clean_signal, ths_percent):
    """
    Calculate adaptive threshold using median standard deviation.
    
    Parameters:
    -----------
    clean_signal : array-like
        Input signal
    ths_percent : float
        Percentage to scale threshold
        
    Returns:
    --------
    float
        Adaptive threshold
    """
    # Calculate the median standard deviation
    baseline_std = np.median(np.std(clean_signal))
    # Scale threshold by a percentage of the baseline standard deviation
    std_ths = baseline_std * (1 + ths_percent / 100)
    return std_ths


# === Advanced Statistical Methods ===

def valley_detection(dataset, fs):
    """
    Detect valleys in PPG signal for segmentation.
    
    Parameters:
    -----------
    dataset : array-like
        PPG signal
    fs : float
        Sampling frequency (Hz)
        
    Returns:
    --------
    list
        Indices of detected valleys
    """
    window = []
    valleylist = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    nvalleys = 0
    valleyarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:
        if (datapoint > localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint <= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            minimum = min(window)
            beatposition = listpos - len(window) + (window.index(min(window)))
            valleylist.append(beatposition)
            window = []
            listpos += 1

    # Ignore if the previous peak was within 360 ms interval
    for val in valleylist:
        if nvalleys > 0:
            prev_valley = valleylist[nvalleys - 1]
            elapsed = val - prev_valley
            if elapsed > TH_elapsed:
                valleyarray.append(val)
        else:
            valleyarray.append(val)
        nvalleys += 1    
    
    print(f"Valleys detected: {valleyarray}")
    return valleyarray


def pair_valley(valley):
    """
    Create pairs of consecutive valleys for segmentation.
    
    Parameters:
    -----------
    valley : list
        List of valley indices
        
    Returns:
    --------
    list
        List of [start, end] pairs
    """
    pair_valley = []
    for i in range(len(valley) - 1):
        pair_valley.append([valley[i], valley[i + 1]])
    return pair_valley


def statistic_detection(signal, fs):
    """
    Calculate statistical features for each signal segment.
    
    Parameters:
    -----------
    signal : array-like
        PPG signal
    fs : float
        Sampling frequency (Hz)
        
    Returns:
    --------
    tuple
        (stds, kurtosis_values, skew_values, valley_pairs)
    """
    valley = pair_valley(valley_detection(signal, fs))
    if len(valley) == 0:
        print("No valleys detected. Returning empty statistics.")
        return [], [], [], valley
    
    stds = [np.std(signal[val[0]:val[1]]) for val in valley]
    kurtosiss = [kurtosis(signal[val[0]:val[1]]) for val in valley]
    skews = [skew(signal[val[0]:val[1]]) for val in valley]

    return stds, kurtosiss, skews, valley


def statistic_threshold(clean_signal, fs, ths):
    """
    Calculate adaptive thresholds based on signal statistics.
    
    Parameters:
    -----------
    clean_signal : array-like
        Input PPG signal
    fs : float
        Sampling frequency (Hz)
    ths : list
        Base threshold values [std_offset, kurt_offset, skew_lower, skew_upper]
        
    Returns:
    --------
    tuple
        (std_threshold, kurt_threshold, skew_thresholds)
    """
    stds, kurtosiss, skews, valley = statistic_detection(clean_signal, fs)

    print(f"Clean Signal Statistics:")
    print(f"Standard Deviations: {stds}")
    print(f"Kurtosis: {kurtosiss}")
    print(f"Skewness: {skews}")

    if not stds or not kurtosiss or not skews:
        print("Warning: No valid statistics. Using fallback thresholds.")
        return ths[0], ths[1], [ths[2], ths[3]]

    std_ths = np.mean(stds) + ths[0]
    kurt_ths = np.mean(kurtosiss) + ths[1]
    skew_ths = [np.mean(skews) - ths[2], np.mean(skews) + ths[3]]

    print(f"Dynamic Thresholds:")
    print(f"Standard Deviation Threshold: {std_ths}")
    print(f"Kurtosis Threshold: {kurt_ths}")
    print(f"Skewness Thresholds: {skew_ths}")

    return std_ths, kurt_ths, skew_ths


def eliminate_noise_in_time(data, fs, ths, cycle=1):
    """
    Eliminate noise using statistical analysis of signal segments.
    
    Parameters:
    -----------
    data : array-like
        PPG signal
    fs : float
        Sampling frequency (Hz)
    ths : list
        Threshold values [std_ths, kurt_ths, [skew_lower, skew_upper]]
    cycle : int, default=1
        Number of cycles to average
        
    Returns:
    --------
    array
        Cleaned signal
    """
    stds, kurtosiss, skews, valley = statistic_detection(data, fs)
    
    # Cycle-averaged statistics
    stds_ = [np.mean(stds[i:i+cycle]) for i in range(0, len(stds) - cycle + 1, cycle)]
    kurtosiss_ = [np.mean(kurtosiss[i:i+cycle]) for i in range(0, len(kurtosiss) - cycle + 1, cycle)]
    skews_ = [np.mean(skews[i:i+cycle]) for i in range(0, len(skews) - cycle + 1, cycle)]
   
    # Extract clean indices
    eli_std = [stds_.index(x) for x in stds_ if x < ths[0]]
    eli_kurt = [kurtosiss_.index(x) for x in kurtosiss_ if x < ths[1]]
    eli_skew = [skews_.index(x) for x in skews_ if x > ths[2][0] and x < ths[2][1]]

    total_list = eli_std + eli_kurt + eli_skew
    
    # Count occurrences
    dic = {}
    for i in total_list:
        dic[i] = dic.get(i, 0) + 1
            
    new_list = [key for key, value in dic.items() if value >= 3]
    new_list.sort()
    
    # Construct eliminated data
    eliminated_data = []
    for x in new_list:
        eliminated_data.extend(data[valley[x * cycle][0]:valley[x * cycle + cycle - 1][1]])
    
    # Convert to numpy array
    eliminated_data = np.array(eliminated_data)
    
    return eliminated_data