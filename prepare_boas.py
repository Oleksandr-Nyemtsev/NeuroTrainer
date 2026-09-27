from pathlib import Path
import csv
import json
import time

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


ROOT = Path("data/external/boas")
OUT = Path("data/processed/boas")
OUT.mkdir(parents=True, exist_ok=True)

FS_OUT = 100
VALID_STAGES = {0, 1, 2, 3, 4}

sos = butter(
    4, [1, 40],
    btype="bandpass",
    fs=256,
    output="sos",
)


def read_events(path):
    with path.open(encoding="utf-8-sig") as f:
        return {
            round(float(row["onset"]), 4): row
            for row in csv.DictReader(f, delimiter="\t")
        }


def prepare_subject(subject_dir):
    subject = subject_dir.name
    eeg_dir = subject_dir / "eeg"

    prefix = f"{subject}_task-Sleep_"

    edf = eeg_dir / f"{prefix}acq-headband_eeg.edf"
    hb_tsv = eeg_dir / f"{prefix}acq-headband_events.tsv"
    psg_tsv = eeg_dir / f"{prefix}acq-psg_events.tsv"

    if not all(p.exists() for p in (edf, hb_tsv, psg_tsv)):
        return {"subject": subject, "status": "missing files"}

    hb = read_events(hb_tsv)
    psg = read_events(psg_tsv)

    raw = mne.io.read_raw_edf(
        edf,
        preload=False,
        include=["HB_1", "HB_2"],
        verbose="ERROR",
    )

    if set(raw.ch_names) != {"HB_1", "HB_2"}:
        return {"subject": subject, "status": "missing EEG channels"}

    fs = raw.info["sfreq"]
    if abs(fs - 256) > 0.01:
        return {
            "subject": subject,
            "status": f"unexpected sampling rate: {fs}",
        }

    X, y, onsets = [], [], []
    rejected_ai = 0
    rejected_signal = 0

    for onset, label in sorted(psg.items()):
        if onset not in hb:
            continue

        stage = int(label["stage_hum"])

        if stage not in VALID_STAGES:
            continue

        if int(hb[onset]["stage_ai"]) == -2:
            rejected_ai += 1
            continue

        start = round(onset * fs)
        stop = start + round(30 * fs)

        if start < 0 or stop > raw.n_times:
            continue

        signal = raw.get_data(
            picks=["HB_1", "HB_2"],
            start=start,
            stop=stop,
        ) * 1e6

        if not np.isfinite(signal).all():
            rejected_signal += 1
            continue

        peak = np.max(np.abs(signal))
        channel_std = signal.std(axis=1)

        if peak > 500 or np.any(channel_std < 0.5):
            rejected_signal += 1
            continue

        signal = sosfiltfilt(sos, signal, axis=-1)
        signal = resample_poly(
            signal, up=25, down=64, axis=-1
        )

        if signal.shape != (2, 3000):
            continue

        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)

        signal = (signal - mean) / np.maximum(std, 1e-6)

        if not np.isfinite(signal).all():
            continue

        X.append(signal.astype(np.float32))
        y.append(stage)
        onsets.append(onset)

    if not X:
        return {"subject": subject, "status": "no usable epochs"}

    output = OUT / f"{subject}.npz"

    np.savez_compressed(
        output,
        X=np.stack(X),
        y=np.asarray(y, dtype=np.int64),
        onsets=np.asarray(onsets, dtype=np.float64),
        subject_id=subject,
        sampling_frequency=FS_OUT,
    )

    return {
        "subject": subject,
        "status": "OK",
        "epochs": len(y),
        "rejected_ai": rejected_ai,
        "rejected_signal": rejected_signal,
        "classes": np.bincount(y, minlength=5).tolist(),
    }


def main():
    start_time = time.time()
    report = []

    subjects = sorted(
        ROOT.glob("sub-*"),
        key=lambda p: p.name,
    )

    print(f"Found {len(subjects)} participant folders.", flush=True)

    for index, subject_dir in enumerate(subjects, start=1):
        try:
            result = prepare_subject(subject_dir)
        except Exception as error:
            result = {
                "subject": subject_dir.name,
                "status": f"ERROR: {error}",
            }

        report.append(result)

        print(
            f"[{index}/{len(subjects)}] "
            f"{result['subject']}: "
            f"{result.get('epochs', 0)} epochs — "
            f"{result['status']}",
            flush=True,
        )

        # Preserve the report even if processing is interrupted.
        (OUT / "processing_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    total = sum(row.get("epochs", 0) for row in report)

    print("\n=== PREPARATION COMPLETE ===")
    print("Recordings processed:", len(report))
    print("Usable epochs:", total)
    print("Minutes:", round((time.time() - start_time) / 60, 1))


if __name__ == "__main__":
    main()
