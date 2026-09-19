import numpy as np
from scipy.signal import welch


def band_power(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapezoid(psd[mask], freqs[mask])


def analyze_channel(signal, fs=256):
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=fs * 2
    )

    delta = band_power(psd, freqs, 1, 4)
    theta = band_power(psd, freqs, 4, 8)
    alpha = band_power(psd, freqs, 8, 13)
    beta = band_power(psd, freqs, 13, 30)
    gamma = band_power(psd, freqs, 30, 40)

    total = delta + theta + alpha + beta + gamma

    if total == 0:
        return {
            "Delta": 0.0,
            "Theta": 0.0,
            "Alpha": 0.0,
            "Beta": 0.0,
            "Gamma": 0.0
        }

    return {
        "Delta": delta / total * 100,
        "Theta": theta / total * 100,
        "Alpha": alpha / total * 100,
        "Beta": beta / total * 100,
        "Gamma": gamma / total * 100
    }