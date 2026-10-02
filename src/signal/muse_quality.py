import numpy as np


class MuseSignalQuality:
    """Conservative quality gate; invalid samples are never filtered."""

    def __init__(self, fs=250):
        self.fs = fs

    def analyze(self, eeg, timestamps=None):
        eeg = np.asarray(eeg, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[0] != 4:
            raise ValueError("Expected EEG shape (4, samples)")
        results = []
        for channel in eeg:
            finite = channel[np.isfinite(channel)]
            missing = 1.0 - len(finite) / len(channel) if len(channel) else 1.0
            amplitude = float(np.percentile(finite, 99) - np.percentile(finite, 1)) if len(finite) else 0.0
            flatline = not len(finite) or np.std(finite) < 0.5
            excessive = amplitude > 300
            valid = missing == 0.0 and not flatline
            quality = (0.5 if excessive else 1.0) if valid else 0.0
            reason = "Invalid samples" if missing else ("Flatline" if flatline else ("Excessive amplitude" if excessive else "OK"))
            results.append({
                "quality": quality,
                "missing_fraction": round(float(missing), 4),
                "amplitude_uv": round(amplitude, 2),
                "flatline": bool(flatline),
                "excessive_amplitude": bool(excessive),
                "reason": reason,
            })
        diagnostic = None
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=float)
            dt = np.diff(ts)
            diagnostic = {
                "negative_intervals": int(np.sum(dt < 0)),
                "zero_intervals": int(np.sum(dt == 0)),
                "large_intervals": int(np.sum(dt > 1.5 / self.fs)),
                "invalid_timestamps": int(np.sum(~np.isfinite(ts))),
            }
        return {"channels": results, "timestamp_diagnostic": diagnostic}
