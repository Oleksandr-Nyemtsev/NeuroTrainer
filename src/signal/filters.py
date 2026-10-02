import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs=256, low=1, high=40, order=4):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("EEG must contain finite samples")
    b, a = butter(
        N=order,
        Wn=[low, high],
        btype="bandpass",
        fs=fs
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal