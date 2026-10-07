"""Rolling Muse observation. No audio or device ownership in this component."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from src.signal.muse_quality import MuseSignalQuality


def prepare_sleep_input(eeg):
    """Muse microvolts, TP9/AF7/AF8/TP10, 256 Hz; experimental BOAS transfer."""
    eeg = np.asarray(eeg, dtype=float)
    if eeg.shape != (4, 7680) or not np.isfinite(eeg).all():
        raise ValueError("Expected finite Muse EEG (4, 7680)")
    selected = eeg[[1, 2]]
    if np.max(np.abs(selected)) > 500 or np.any(selected.std(-1) < .5):
        raise ValueError("Unusable frontal EEG")
    sos = butter(4, [1, 40], btype="bandpass", fs=256, output="sos")
    x = resample_poly(sosfiltfilt(sos, selected, axis=-1), 25, 64, axis=-1)
    std = x.std(-1, keepdims=True)
    if np.any(std < 1e-6):
        raise ValueError("Flat filtered EEG")
    return ((x - x.mean(-1, keepdims=True)) / std).astype(np.float32)


class SleepObserver:
    def __init__(self, model, fs=256):
        if fs != 256:
            raise ValueError("This Muse adapter requires 256 Hz")
        self.model = model
        self.quality = MuseSignalQuality(fs=fs)
        self.reset()

    def reset(self):
        self.buffer = np.empty((4, 0))
        self.timestamps = np.empty(0)
        self.last_prediction = None
        self.result = {"ready": False, "reason": "Collecting 30 seconds"}

    def update(self, eeg, timestamps):
        eeg, ts = np.asarray(eeg), np.asarray(timestamps)
        if (eeg.ndim != 2 or eeg.shape[0] != 4 or ts.ndim != 1
                or eeg.shape[1] != len(ts) or not len(ts)
                or not np.isfinite(eeg).all() or not np.isfinite(ts).all()):
            self.reset()
            raise ValueError("Invalid observer input")
        times = np.r_[self.timestamps[-1:], ts]
        dt = np.diff(times)
        if np.any(dt <= 0) or np.any(dt > 1.5/256):
            self.reset()
            raise ValueError("Observer timestamp gap")
        scores = self.quality.analyze(eeg)["channels"]
        if min(c["quality"] for c in scores) < .75 or np.max(np.abs(eeg[[1,2]])) > 500:
            # Explain each independent gate before resetting the rolling context.
            details = []
            for i, (name, channel, quality) in enumerate(zip(
                    ("TP9", "AF7", "AF8", "TP10"), eeg, scores)):
                peak = float(np.max(np.abs(channel)))
                reasons = []
                if quality["quality"] < .75:
                    reasons.append(quality["reason"] + " (quality < 0.75)")
                if i in (1, 2) and peak > 500:
                    reasons.append("absolute peak > 500 uV")
                details.append(
                    f"{name}: mean={channel.mean():.2f} uV, "
                    f"std={channel.std():.2f} uV, "
                    f"p99-p1={quality['amplitude_uv']:.2f} uV, "
                    f"abs_peak={peak:.2f} uV, "
                    f"reason={'; '.join(reasons) if reasons else 'OK'}"
                )
            self.reset()
            raise ValueError("Unreliable EEG for sleep observation | " + " | ".join(details))
        self.buffer = np.concatenate([self.buffer, eeg], axis=1)[:, -7680:]
        self.timestamps = np.r_[self.timestamps, ts][-7680:]
        if self.buffer.shape[1] < 7680:
            self.result = {"ready": False, "reason": "Collecting 30 seconds",
                           "buffer_seconds": self.buffer.shape[1]/256}
            return self.result
        if self.last_prediction is not None and ts[-1] - self.last_prediction < 2 - 1e-6:
            return self.result
        probs = self.model.predict(prepare_sleep_input(self.buffer))
        values = np.asarray(list(probs.values()), dtype=float)
        if (set(probs) != {"Wake", "N1", "N2", "N3", "REM"}
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=1e-5)):
            self.reset()
            raise ValueError("Invalid sleep probabilities")
        self.last_prediction = float(ts[-1])
        self.result = {"ready": True, "probabilities": probs,
                       "stage": max(probs, key=probs.get),
                       "window_start": float(self.timestamps[0]),
                       "window_end": float(ts[-1]), "experimental_muse_transfer": True}
        return self.result
