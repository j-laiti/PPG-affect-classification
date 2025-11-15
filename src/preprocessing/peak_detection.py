"""
PPG Peak Detection Module

Implements threshold-based peak detection for PPG signals using local averaging
and minimum inter-peak interval constraints. Adapted from Heo et al. (2021).

Reference: https://doi.org/10.1109/ACCESS.2021.3060441

Author: Justin Laiti (adapted from Heo et al., 2021)
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

#%% Imports
import numpy as np
from scipy.signal import find_peaks
from .filters import moving_average_filter

#%% Peak Detection

def threshold_peakdetection(dataset, fs):
    """
    Detect PPG signal peaks using threshold-based local maximum detection.
    
    Identifies peaks as local maxima above the signal's global average,
    then filters based on minimum physiological inter-peak interval.
    
    Args:
        dataset: PPG signal array
        fs: Sampling frequency in Hz
        
    Returns:
        peakarray: List of filtered peak indices
        
    Note:
        Minimum inter-peak interval is set to 0.45s (~133 BPM max HR)
        to exclude physiologically implausible peaks.
    """
    dataset = dataset.copy()
    window = []
    peaklist = []
    localaverage = np.average(dataset)
    
    # Minimum samples between peaks (0.45s = ~133 BPM max)
    TH_elapsed = np.ceil(0.45 * fs)
    
    # Step 1: Identify all candidate peaks above local average
    for listpos, datapoint in enumerate(dataset):
        if datapoint < localaverage and len(window) < 1:
            continue
        elif datapoint >= localaverage:
            window.append(datapoint)
        else:
            # Found local maximum - add to candidate list
            maximum = max(window)
            beatposition = listpos - len(window) + window.index(maximum)
            peaklist.append(beatposition)
            window = []

    # Step 2: Filter peaks based on minimum inter-peak interval
    peakarray = []
    for val in peaklist:
        if len(peakarray) > 0:
            elapsed = val - peakarray[-1]
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)  # Always keep first peak
    
    return peakarray