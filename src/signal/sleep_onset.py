import numpy as np


def find_stable_n2(stages, required_epochs=3):
    """
    Find the first sequence of consecutive N2 epochs.

    Stage labels:
    0 = Wake
    1 = N1
    2 = N2
    3 = N3
    4 = REM

    Returns the index of the first epoch
    in the stable N2 sequence, or None.
    """

    if required_epochs < 1:
        raise ValueError("required_epochs must be positive")
    consecutive = 0

    for index, stage in enumerate(stages):

        if stage == 2:
            consecutive += 1

            if consecutive >= required_epochs:
                return index - required_epochs + 1

        else:
            consecutive = 0

    return None


ISRUC_ONSET_RULE = "isruc_stable_n2_3_epochs"
MUSE_ONSET_RULE = "muse_n2_onset_annotation"
ONSET_FIELDS = ["subject", "recording", "onset", "window_start", "window_end",
                "n2_onset", "stage", "time_to_n2_seconds", "label_definition", "eeg_epoch_index"]


def make_onset_row(subject, recording, window_start, window_end, n2_onset,
                   label_definition, stage=None):
    times = np.asarray([window_start, window_end, n2_onset], dtype=float)
    if not np.isfinite(times).all() or window_start < 0 or window_end <= window_start:
        raise ValueError("Invalid onset/window timestamps")
    if window_end > n2_onset:
        raise ValueError("Window extends beyond the labeled N2 onset")
    if label_definition not in (ISRUC_ONSET_RULE, MUSE_ONSET_RULE):
        raise ValueError("Specify the dataset's onset definition explicitly")
    return {"subject": subject, "recording": recording, "onset": float(window_start),
            "window_start": float(window_start), "window_end": float(window_end),
            "n2_onset": float(n2_onset), "stage": stage,
            "time_to_n2_seconds": float(n2_onset - window_end),
            "label_definition": label_definition}


def align_onset_rows(npz_path, rows):
    """Resolve EEG indices by recording identity and time, never by CSV position.

    Missing timestamps fail explicitly; callers may report a skipped recording.
    This function only reads NPZ metadata/labels and never rewrites EEG files.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"onsets", "window_start", "window_end", "recording", "subject_id", "y"}
        if not required.issubset(data.files):
            raise ValueError("NPZ lacks alignment metadata; regenerate it with the fixed prepare_isruc.py")
        starts = data["window_start"]
        ends = data["window_end"]
        onsets = data["onsets"]
        labels = data["y"]
        subject = str(data["subject_id"].item())
        recording = str(data["recording"].item())
        if (starts.ndim != 1 or ends.shape != starts.shape or onsets.shape != starts.shape
                or len(labels) != len(starts) or not np.isfinite(starts).all()
                or not np.isfinite(ends).all() or not np.isfinite(onsets).all()
                or not np.allclose(starts, onsets, rtol=0, atol=1e-6)
                or np.any(ends <= starts) or len(np.unique(starts)) != len(starts)):
            raise ValueError("Invalid or duplicate NPZ timestamps")
        aligned = []
        used = set()
        for row in rows:
            if row["subject"] != subject or row["recording"] != recording:
                raise ValueError("EEG and label recording identities differ")
            matches = np.flatnonzero(np.isclose(starts, row["window_start"], rtol=0, atol=1e-6)
                                     & np.isclose(ends, row["window_end"], rtol=0, atol=1e-6))
            if len(matches) != 1:
                raise ValueError(f"No unique EEG window for start={row['window_start']}")
            index = int(matches[0])
            if index in used:
                raise ValueError("Duplicate label window")
            if row["stage"] is not None and row["stage"] != int(labels[index]):
                raise ValueError("EEG and onset labels disagree")
            used.add(index)
            aligned.append({**row, "eeg_epoch_index": index})
        return aligned


def subject_split_lookup(split):
    lookup = {}
    for group in ("train", "val", "test"):
        for subject in split[group]:
            if subject in lookup:
                raise ValueError("Subject appears more than once in splits")
            lookup[subject] = group
    return lookup
