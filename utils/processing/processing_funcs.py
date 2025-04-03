# apply the moving average and other filters here

# Preprocessing method based on this paper:
# S. Heo, S. Kwon and J. Lee, "Stress Detection With Single PPG Sensor by Orchestrating Multiple Denoising and Peak-Detecting Methods," 
# in IEEE Access, vol. 9, pp. 47777-47785, 2021, doi: 10.1109/ACCESS.2021.3060441.
# Adapted March 3rd 2025 by Justin Laiti

import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter

#%% Filters ##

# bandpass filter

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

# original moving average

def moving_average_filter(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# simple noise elimination with custom threshold

def simple_dynamic_threshold(clean_signal, percentile, window_size = 3):
    """
    Calculate the dynamic threshold based on the 95th percentile of standard deviations
    across signal segments.
    """
    segment_stds = [
        np.std(clean_signal[i:i + window_size])
        for i in range(0, len(clean_signal) - window_size, window_size)
    ]
    threshold = np.percentile(segment_stds, percentile)
    return threshold

def simple_noise_elimination(clean_signal, fs, threshold, window_size=3):
    """
    Eliminate noisy segments of the signal based on the calculated threshold.

    Parameters:
    - clean_signal: The input signal to clean.
    - fs: Sampling frequency of the signal.
    - threshold: The standard deviation threshold for noise elimination.
    - window_size: The size of the window (in seconds) to evaluate noise.

    Returns:
    - clean_signal_filtered: The cleaned signal with noisy segments removed.
    - clean_indices: The indices of the clean segments in the original signal.
    """
    clean_indices = []
    step_size = int(window_size * fs)  # Convert window size to samples

    for i in range(0, len(clean_signal), step_size):
        segment = clean_signal[i:i + step_size]
        if np.std(segment) < threshold:
            clean_indices.extend(range(i, i + len(segment)))

    # Convert clean_indices to a NumPy array for indexing
    clean_indices = np.array(clean_indices)

    # Return the cleaned signal and the indices
    clean_signal_filtered = clean_signal[clean_indices]
    return clean_signal_filtered, clean_indices

# simple noise elimination with adaptive threshold and median instead of mean

def simple_statistic_adaptive_threshold(clean_signal, ths_percent):
    # Calculate the median standard deviation
    baseline_std = np.median(np.std(clean_signal))
    # Scale threshold by a percentage of the baseline standard deviation
    std_ths = baseline_std * (1 + ths_percent / 100)
    return std_ths

# Define the statistical noise elimination function

def statistic_threshold(clean_signal, fs, ths):
    stds, kurtosiss, skews, valley = statistic_detection(clean_signal, fs)

    print(f"Clean Signal Statistics:")
    print(f"Standard Deviations: {stds}")
    print(f"Kurtosis: {kurtosiss}")
    print(f"Skewness: {skews}")

    if not stds or not kurtosiss or not skews:
        print("Warning: No valid statistics. Using fallback thresholds.")
        return ths[0], ths[1], [ths[2], ths[3]]  # Return user-provided thresholds

    std_ths = np.mean(stds) + ths[0]
    kurt_ths = np.mean(kurtosiss) + ths[1]
    skew_ths = [np.mean(skews) - ths[2], np.mean(skews) + ths[3]]

    print(f"Dynamic Thresholds:")
    print(f"Standard Deviation Threshold: {std_ths}")
    print(f"Kurtosis Threshold: {kurt_ths}")
    print(f"Skewness Thresholds: {skew_ths}")

    return std_ths, kurt_ths, skew_ths


def statistic_detection(signal, fs):
    valley = pair_valley(valley_detection(signal, fs))
    if len(valley) == 0:
        print("No valleys detected. Returning empty statistics.")
        return [], [], [], valley
    
    stds = [np.std(signal[val[0]:val[1]]) for val in valley]
    kurtosiss = [kurtosis(signal[val[0]:val[1]]) for val in valley]
    skews = [skew(signal[val[0]:val[1]]) for val in valley]

    return stds, kurtosiss, skews, valley

# Define the statistical noise elimination function
def eliminate_noise_in_time(data, fs, ths, cycle=1):
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


# +
def valley_detection(dataset, fs):
    window = []
    valleylist = []
    ybeat = []
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

    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
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
    pair_valley=[]
    for i in range(len(valley)-1):
        pair_valley.append([valley[i], valley[i+1]])
    return pair_valley

def standardize(ppg_signal):
    mean = ppg_signal.mean()
    std = ppg_signal.std()
    if std == 0:  # Prevent division by zero
        return np.zeros_like(ppg_signal)
    norm_signal = (ppg_signal - mean) / std
    return norm_signal