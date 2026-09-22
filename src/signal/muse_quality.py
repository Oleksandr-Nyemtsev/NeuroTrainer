
import numpy as np


class MuseSignalQuality:
    """Basic EEG quality and timestamp diagnostics."""

    def __init__(self, fs=250):
        self.fs = fs

    def analyze(self, eeg, timestamps=None):
        eeg = np.asarray(eeg, dtype=np.float64)

        if eeg.ndim != 2 or eeg.shape[0] != 4:
            raise ValueError("Expected EEG shape (4, samples)")

        results = []

        for channel in eeg:
            finite = channel[np.isfinite(channel)]

            if len(finite) == 0:
                results.append({
                    "quality": 0.0,
                    "reason": "No valid samples",
                })
                continue

            missing = 1 - len(finite) / len(channel)
            amplitude = np.percentile(finite, 99) - np.percentile(finite, 1)

            # Preliminary thresholds, not clinically validated.
            excessive_amplitude = amplitude > 300
            flatline = np.std(finite) < 0.5

            quality = 1.0 - missing

            if excessive_amplitude:
                quality *= 0.5

            if flatline:
                quality = 0.0

            results.append({
                "quality": round(float(quality), 3),
                "missing_fraction": round(float(missing), 4),
                "amplitude_uv": round(float(amplitude), 2),
                "flatline": bool(flatline),
                "excessive_amplitude": bool(excessive_amplitude),
            })

        diagnostic = None

        if timestamps is not None:
            dt = np.diff(np.asarray(timestamps, dtype=float))

            diagnostic = {
                "negative_intervals": int(np.sum(dt < 0)),
                "zero_intervals": int(np.sum(dt == 0)),
                "large_intervals": int(
                    np.sum(dt > 1.5 / self.fs)
                ),
            }

        return {
            "channels": results,
            "timestamp_diagnostic": diagnostic,
        }