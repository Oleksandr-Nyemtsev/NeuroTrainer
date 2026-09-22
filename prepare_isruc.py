
from pathlib import Path
import argparse
import csv

import mne
import numpy as np
import pandas as pd


SOURCE = Path("data/external/isruc/nm000111")
OUTPUT = Path("data/processed/isruc_v2")

TARGET_FS = 100
EPOCH_SECONDS = 30
WAKE_MARGIN = 60

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

# Keep electrode configurations distinguishable.
CHANNEL_CONFIGS = [
    ("M2", ["C3-M2", "O1-M2"]),
    ("A2", ["C3-A2", "O1-A2"]),
]


def select_eeg(raw):
    available = set(raw.ch_names)

    for scheme, channels in CHANNEL_CONFIGS:
        if all(ch in available for ch in channels):
            raw.pick(channels)
            raw.load_data(verbose=False)

            return (
                raw.get_data(),
                scheme,
                channels,
            )

    # Some EDF files store electrodes separately.
    # Construct the same referenced channels explicitly.
    if {"C3", "O1", "A2"}.issubset(available):
        channels = ["C3", "O1", "A2"]

        raw.pick(channels)
        raw.load_data(verbose=False)

        data = raw.get_data()
        index = {
            name: i
            for i, name in enumerate(raw.ch_names)
        }

        c3_a2 = data[index["C3"]] - data[index["A2"]]
        o1_a2 = data[index["O1"]] - data[index["A2"]]

        eeg = np.stack([c3_a2, o1_a2])

        return (
            eeg,
            "A2_derived",
            channels,
        )

    raise ValueError(
        f"No supported electrode pair. "
        f"Available: {raw.ch_names}"
    )


def prepare_recording(edf_file):
    prefix = edf_file.name.removesuffix("_eeg.edf")

    events_file = edf_file.with_name(
        prefix + "_events.tsv"
    )

    if not events_file.exists():
        raise FileNotFoundError(
            f"Missing events: {events_file}"
        )

    # A subject can have two sessions.
    subject = next(
        parent.name
        for parent in edf_file.parents
        if parent.name.startswith("sub-")
    )

    session = next(
        (
            parent.name
            for parent in edf_file.parents
            if parent.name.startswith("ses-")
        ),
        "ses-1",
    )

    print(f"\nProcessing {subject} {session}")

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False,
    )

    try:
        original_fs = float(raw.info["sfreq"])

        eeg, scheme, source_channels = select_eeg(raw)

    finally:
        raw.close()

    # Resample with antialias filtering.
    if original_fs != TARGET_FS:
        from scipy.signal import resample_poly
        from fractions import Fraction

        ratio = Fraction(
            TARGET_FS / original_fs
        ).limit_denominator(10000)

        eeg = resample_poly(
            eeg,
            ratio.numerator,
            ratio.denominator,
            axis=1,
        )

    events = pd.read_csv(
        events_file,
        sep="\t",
    )

    events["onset"] = pd.to_numeric(
        events["onset"],
        errors="coerce",
    )

    events["duration"] = pd.to_numeric(
        events["duration"],
        errors="coerce",
    )

    valid = events[
        events["trial_type"].isin(LABEL_MAP)
        & np.isclose(
            events["duration"],
            EPOCH_SECONDS,
        )
    ].sort_values("onset").copy()

    # Do not silently use duplicate labels for one epoch.
    if valid["onset"].duplicated().any():
        raise ValueError(
            "Duplicate sleep-stage annotations at the same onset"
        )

    if valid.empty:
        raise ValueError("No valid sleep-stage annotations")

    labels = valid["trial_type"].map(
        LABEL_MAP
    ).to_numpy(dtype=np.int64)

    sleep_indices = np.flatnonzero(labels != 0)

    if len(sleep_indices) == 0:
        raise ValueError("No sleep epochs")

    start = max(
        0,
        sleep_indices[0] - WAKE_MARGIN,
    )

    stop = min(
        len(valid),
        sleep_indices[-1] + WAKE_MARGIN + 1,
    )

    valid = valid.iloc[start:stop]
    labels = labels[start:stop]

    samples = TARGET_FS * EPOCH_SECONDS

    X_parts = []
    y_parts = []
    skipped = 0

    for (_, event), label in zip(
        valid.iterrows(),
        labels,
    ):
        sample = round(
            float(event["onset"]) * TARGET_FS
        )

        epoch = eeg[
            :,
            sample:sample + samples,
        ]

        if (
            epoch.shape != (2, samples)
            or not np.isfinite(epoch).all()
        ):
            skipped += 1
            continue

        mean = epoch.mean(
            axis=1,
            keepdims=True,
        )

        std = epoch.std(
            axis=1,
            keepdims=True,
        )

        # Same normalization as our Sleep-EDF pipeline.
        epoch = (
            (epoch - mean) / (std + 1e-8)
        ).astype(np.float32)

        X_parts.append(epoch)
        y_parts.append(label)

    if not X_parts:
        raise ValueError("No usable epochs")

    X = np.stack(X_parts)
    y = np.asarray(
        y_parts,
        dtype=np.int64,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT / f"{prefix}.npz"

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        subject_id=subject,
        session_id=session,
        channel_scheme=scheme,
        source_channels=np.asarray(
            source_channels,
        ),
        sampling_frequency=TARGET_FS,
    )

    counts = np.bincount(
        y,
        minlength=5,
    ).tolist()

    print(
        f"SAVED {output_file.name} | "
        f"scheme={scheme} | "
        f"epochs={len(y)} | "
        f"classes={counts} | "
        f"skipped={skipped}"
    )

    return {
        "recording": prefix,
        "subject": subject,
        "session": session,
        "scheme": scheme,
        "epochs": len(y),
        "skipped": skipped,
        "status": "OK",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        default=None,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    recordings = sorted(
        SOURCE.rglob("*_eeg.edf")
    )

    if not args.all:
        if not args.subject:
            parser.error(
                "Use --subject I040 or --all"
            )

        subject_name = f"sub-{args.subject}"

        recordings = [
            f for f in recordings
            if subject_name in f.parts
        ]

    print(
        f"Found {len(recordings)} recordings"
    )

    report = []

    for index, edf_file in enumerate(
        recordings,
        start=1,
    ):
        print(
            f"\n[{index}/{len(recordings)}] "
            f"{edf_file.name}"
        )

        try:
            result = prepare_recording(
                edf_file
            )

        except Exception as exc:
            print(
                f"FAILED: {edf_file.name}: {exc}"
            )

            result = {
                "recording": edf_file.stem,
                "subject": "",
                "session": "",
                "scheme": "",
                "epochs": 0,
                "skipped": 0,
                "status": "FAILED",
                "error": str(exc),
            }

        report.append(result)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = OUTPUT / "processing_report.csv"

    with report_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "recording",
                "subject",
                "session",
                "scheme",
                "epochs",
                "skipped",
                "status",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(report)

    successful = sum(
        item["status"] == "OK"
        for item in report
    )

    print(
        f"\nCOMPLETE: "
        f"{successful}/{len(report)} recordings"
    )
    print(
        f"Report: {report_file}"
    )


if __name__ == "__main__":
    main()