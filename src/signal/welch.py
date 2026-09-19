from scipy.signal import welch


def compute_welch_psd(signal, fs=256, window_seconds=2):
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=fs * window_seconds
    )

    return freqs, psd