
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


class MuseDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def list_recordings(self):
        return sorted(self.data_dir.glob("EEG_*.csv"))

    def load(self, filename):
        path = self.data_dir / filename
        df = pd.read_csv(path)

        required = ["timestamps", *CHANNELS]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        timestamps = df["timestamps"].to_numpy(dtype=np.float64)
        eeg = df[CHANNELS].to_numpy(dtype=np.float64).T

        # Validate before estimating the sampling frequency.
        dt = np.diff(timestamps)
        valid_dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(valid_dt) == 0:
            raise ValueError("Invalid timestamps")

        fs = 1.0 / np.median(valid_dt)

        return {
            "eeg": eeg,
            "timestamps": timestamps,
            "fs": fs,
            "channels": CHANNELS,
            "markers": df["Marker0"].to_numpy()
            if "Marker0" in df.columns else None,
        }