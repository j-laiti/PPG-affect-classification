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