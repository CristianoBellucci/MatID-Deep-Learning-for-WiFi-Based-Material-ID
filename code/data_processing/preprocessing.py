"""Signal preprocessing utilities for WiFi-based Material Identification."""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy import spatial
import torch

from config import DEFAULT_WINDOW_SIZE, DEFAULT_SAMPLINGS, DEFAULT_RESAMPLING_RATIO, EPSILON

def find_indices(list_to_check, item_to_find):
    """Find all indices of an item in a list.
    
    Args:
        list_to_check: Input list to search in
        item_to_find: The item to find in the list
        
    Returns:
        List of indices where the item was found
    """
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def calculate_median(arr):
    """Calculate the median of each array in the input array.
    
    Args:
        arr: Array of arrays whose medians to compute
        
    Returns:
        Array of medians
    """
    return np.array([np.median(elem) for elem in arr])

def phase_sanitization(csi_phases):
    """Perform phase calibration on CSI phase data.
    
    Args:
        csi_phases: Array of phase values to sanitize
        
    Returns:
        Array of sanitized phase values
    """
    all_phase_san = []
    for phases in csi_phases:
        phases_san = []
        for i, phase in enumerate(phases, start=1):
            b = (1/i) * sum(phases[:i])  # phase

            if i == 1:
                phase_san = phase - b
                phases_san.append(phase_san)
            else:
                a = (phase - phases[0]) / (i - 1)  # slope
                phase_san = phase - a*i - b
                phases_san.append(phase_san)
        all_phase_san.append(phases_san)

    return np.asarray(all_phase_san)

def compute_stft(signal, fs=1, stride=1, wind_wid=5, dft_wid=5, window_type='gaussian'):
    """Compute the Short Time Fourier Transform (STFT) of a signal.
    
    Args:
        signal: Time series of measurement values
        fs: Sampling frequency
        stride: Stride of the window
        wind_wid: Width of the window
        dft_wid: Size of the DFT
        window_type: Type of window to use ('gaussian' or 'rect')
        
    Returns:
        f_bins: Frequency bins
        stft_spectrum: STFT of the input signal
    """
    if window_type == 'gaussian':
        window = signal.windows.gaussian(wind_wid, (wind_wid-1)/np.sqrt(8*np.log(200)), sym=True)
    elif window_type == 'rect':
        window = np.ones((wind_wid,))
    else:
        window = signal.get_window(window_type, wind_wid)

    f_bins, _, stft_spectrum = signal.stft(
        x=signal, 
        fs=fs, 
        window=window, 
        nperseg=wind_wid, 
        noverlap=wind_wid-stride, 
        nfft=dft_wid,
        axis=-1, 
        detrend=False, 
        return_onesided=False, 
        boundary='zeros', 
        padded=True
    )

    # Take the absolute value of the STFT spectrum to get magnitude
    stft_spectrum = np.abs(stft_spectrum)
    stft_spectrum = stft_spectrum.reshape(1068, -1)
    return f_bins, stft_spectrum

def normalize_data(data):
    """Normalize input data to have zero mean and unit variance.
    
    Args:
        data: Input data to normalize
        
    Returns:
        Normalized data
    """
    return (data - np.mean(data)) / np.std(data)

def amplitude_sanitization(csi_amplitudes, eps=EPSILON):
    """Sanitize amplitude values by removing outliers.
    
    Args:
        csi_amplitudes: Array of amplitude values to sanitize
        eps: Threshold for identifying outliers
        
    Returns:
        Array of sanitized amplitude values
    """
    normalized_amplitude = []
    
    for amplitudes in csi_amplitudes:
        sanitized_amplitudes = []
        for i in range(len(amplitudes)):
            if i == 0:
                sanitized_amplitudes.append(amplitudes[i])
            else:
                diff = amplitudes[i] - amplitudes[i-1]
                if abs(diff) > eps:
                    sanitized_amplitudes.append(sanitized_amplitudes[-1])
                else:
                    sanitized_amplitudes.append(amplitudes[i])
        normalized_amplitude.append(sanitized_amplitudes)
    
    return np.array(normalized_amplitude)

def csi_to_spectrogram(csis, window_size=DEFAULT_WINDOW_SIZE, 
                      samplings=DEFAULT_SAMPLINGS, resampling_ratio=DEFAULT_RESAMPLING_RATIO):
    """Convert CSI data to spectrogram.
    
    Args:
        csis: Input CSI data
        window_size: Size of the window for STFT
        samplings: Number of time instants
        resampling_ratio: Resampling ratio
        
    Returns:
        Spectrogram of the input CSI data
    """
    # Implementation of csi_to_spec function
    # ... (to be implemented based on the original code)
    pass

def prepare_spectrogram_for_resnet(csis, hann_window=DEFAULT_WINDOW_SIZE, 
                                  samplings=DEFAULT_SAMPLINGS, resampling_ratio=DEFAULT_RESAMPLING_RATIO):
    """Prepare spectrogram for ResNet input.
    
    Args:
        csis: Input CSI data
        hann_window: Size of the Hann window
        samplings: Number of time instants
        resampling_ratio: Resampling ratio
        
    Returns:
        Spectrogram resized for ResNet input
    """
    # Implementation of sanified_spectrogram_alt function
    # ... (to be implemented based on the original code)
    pass
