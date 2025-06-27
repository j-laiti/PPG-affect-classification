import numpy as np
from scipy.signal import find_peaks
from preprocessing.filters import moving_average_filter

#%% Peak detection functions

def threshold_peakdetection(dataset, fs):
    dataset = dataset.copy()  # Ensure the input signal is not modified
    window = []
    peaklist = []
    localaverage = np.average(dataset)
    TH_elapsed = np.ceil(0.45 * fs)  # Threshold for elapsed time between peaks in samples
    
    # Step 1: Identify all candidate peaks
    for listpos, datapoint in enumerate(dataset):
        if datapoint < localaverage and len(window) < 1:
            continue
        elif datapoint >= localaverage:
            window.append(datapoint)
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + window.index(maximum)
            peaklist.append(beatposition)
            window = []

    # Step 2: Filter peaks based on TH_elapsed
    peakarray = []
    for val in peaklist:
        if len(peakarray) > 0:  # Check if there are any peaks in peakarray
            prev_peak = peakarray[-1]  # Use the last filtered peak
            elapsed = val - prev_peak
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)  # Always add the first peak
    
    return peakarray  # Return the filtered peak array

