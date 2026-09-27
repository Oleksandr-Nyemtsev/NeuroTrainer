from pathlib import Path
import csv

import mne
import numpy as np


DATA_DIR = Path("data/external/boas")
VALID_STAGES = {0, 1, 2, 3, 4}


def load_boas_subject(subject="sub-1"):
    eeg_dir = DATA_DIR / subject / "eeg"

    eeg_file = (
        eeg_dir
        / f"{subject}_task-Sleep_acq-headband_eeg.edf"
    )

    labels_file = (
        eeg_dir
        / f"{subject}_task-Sleep_acq-psg_events.tsv"
    )

    # Read human-scored PSG labels.
    with labels_file.open(encoding="utf-8-sig") as file:
        annotations = list(
            csv.DictReader(file, delimiter="\t")
        )

    # Load wearable EEG, not the laboratory PSG signal.
    raw = mne.io.read_raw_edf(
        eeg_file,
        preload=False,
        verbose=False,
    )

    print("Channels:", raw.ch_names)
    print("Sampling rate:", raw.info["sfreq"])

    fs = raw.info["sfreq"]
    epoch_samples = round(30 * fs)

    X = []
    y = []

    for row in annotations:
        stage = int(row["stage_hum"])

        if stage not in VALID_STAGES:
            continue

        onset = float(row["onset"])
        start = round(onset * fs)
        stop = start + epoch_samples

        if stop > raw.n_times:
            continue

        # EDF values are returned in volts.
        # Convert to microvolts.
        signal = raw.get_data(
            picks=["HB_1", "HB_2"],
            start=start,
            stop=stop,
        ) * 1e6

        if not np.isfinite(signal).all():
            continue

        X.append(signal.astype(np.float32))
        y.append(stage)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)

    return X, y


if __name__ == "__main__":
    X, y = load_boas_subject("sub-1")

    print("EEG shape:", X.shape)
    print("Labels shape:", y.shape)

    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    print("Stage counts:", dict(zip(unique, counts)))
