from pathlib import Path

import mne
import numpy as np


ANNOTATION_DESC_TO_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}


LABEL_MAP = {
    1: 0,  # Wake
    2: 1,  # N1
    3: 2,  # N2
    4: 3,  # N3
    5: 4,  # REM
}


def preprocess_sleep_edf_recording(
    psg_file,
    hypnogram_file,
    wake_margin_epochs=60,
):

    print(f"Reading: {Path(psg_file).name}")

    raw = mne.io.read_raw_edf(
        psg_file,
        preload=True,
        verbose=False,
    )

    annotations = mne.read_annotations(
        hypnogram_file,
    )

    raw.set_annotations(
        annotations,
    )

    raw.set_channel_types(
        {
            "EEG Fpz-Cz": "eeg",
            "EEG Pz-Oz": "eeg",
            "EOG horizontal": "eog",
            "Resp oro-nasal": "misc",
            "EMG submental": "emg",
            "Temp rectal": "misc",
            "Event marker": "misc",
        },
        on_unit_change="ignore",
    )

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=ANNOTATION_DESC_TO_EVENT_ID,
        chunk_duration=30.0,
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=30 - 1 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    eeg_picks = mne.pick_types(
        epochs.info,
        eeg=True,
        eog=False,
        emg=False,
        misc=False,
    )

    X = epochs.get_data(
        picks=eeg_picks,
    )

    y = epochs.events[:, 2]

    sleep_indices = np.where(
        y != 1
    )[0]

    if len(sleep_indices) == 0:
        raise ValueError(
            "No sleep epochs found."
        )

    first_sleep = sleep_indices[0]
    last_sleep = sleep_indices[-1]

    start = max(
        0,
        first_sleep - wake_margin_epochs,
    )

    stop = min(
        len(y),
        last_sleep + wake_margin_epochs + 1,
    )

    X = X[start:stop]
    y = y[start:stop]

    y = np.array(
        [
            LABEL_MAP[label]
            for label in y
        ],
        dtype=np.int64,
    )

    X_mean = X.mean(
        axis=2,
        keepdims=True,
    )

    X_std = X.std(
        axis=2,
        keepdims=True,
    )

    X = (
        X - X_mean
    ) / (
        X_std + 1e-8
    )

    X = X.astype(
        np.float32,
    )

    return X, y