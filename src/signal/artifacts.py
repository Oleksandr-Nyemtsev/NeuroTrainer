import numpy as np


def detect_artifact(signal, threshold_uv=150):
    peak_to_peak = np.ptp(signal)

    if peak_to_peak > threshold_uv:
        return True

    return False