
from pathlib import Path
import csv
import json

from src.signal.sleep_onset import (
    find_stable_n2, make_onset_row, align_onset_rows, subject_split_lookup,
    ONSET_FIELDS, ISRUC_ONSET_RULE,
)


SOURCE = Path("data/external/isruc/nm000111")
PROCESSED = Path("data/processed/isruc_v2")
SPLIT_FILE = Path("models/isruc_training/subject_split.json")
OUTPUT = Path("data/processed/sleep_onset")

STAGES = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

FIELDS = ONSET_FIELDS


def process_recording(edf_path):
    prefix = edf_path.name.removesuffix("_eeg.edf")
    events_path = edf_path.with_name(prefix + "_events.tsv")

    subject = next(
        part for part in edf_path.parts
        if part.startswith("sub-")
    )

    with events_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        events = list(
            csv.DictReader(handle, delimiter="\t")
        )

    valid = []

    for event in events:
        if event["trial_type"] not in STAGES:
            continue

        onset = float(event["onset"])
        duration = float(event["duration"])

        if abs(duration - 30.0) > 0.01:
            continue

        valid.append({
            "onset": onset,
            "stage": STAGES[event["trial_type"]],
        })

    valid.sort(key=lambda item: item["onset"])

    stages = [item["stage"] for item in valid]

    stable_index = find_stable_n2(
        stages,
        required_epochs=3,
    )

    if stable_index is None:
        raise ValueError("No stable N2")

    stable_time = valid[stable_index]["onset"]

    # Verify the three N2 epochs are consecutive in time.
    stable_times = [
        valid[i]["onset"]
        for i in range(stable_index, stable_index + 3)
    ]

    if any(
        abs(b - a - 30.0) > 0.01
        for a, b in zip(stable_times, stable_times[1:])
    ):
        raise ValueError("Gap inside stable N2")

    before_n2 = valid[:stable_index]

    if not before_n2:
        raise ValueError("No pre-N2 epochs")

    # Do not silently train on an incomplete timeline.
    timeline = [
        item["onset"] for item in before_n2
    ] + [stable_time]

    if any(
        abs(b - a - 30.0) > 0.01
        for a, b in zip(timeline, timeline[1:])
    ):
        raise ValueError("Gap in pre-N2 timeline")

    rows = []

    for item in before_n2:
        rows.append(make_onset_row(
            subject, prefix, item["onset"], item["onset"] + 30.0,
            stable_time, ISRUC_ONSET_RULE, stage=item["stage"],
        ))

    return subject, rows


def main():
    with SPLIT_FILE.open(encoding="utf-8") as handle:
        split = json.load(handle)

    subject_to_split = subject_split_lookup(split)

    recordings = sorted(SOURCE.rglob("*_eeg.edf"))

    if len(recordings) != 126:
        raise ValueError(
            f"Expected 126 recordings, found {len(recordings)}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = {
        "train": [],
        "val": [],
        "test": [],
    }

    report = []
    successful = 0

    for index, edf_path in enumerate(recordings, 1):
        prefix = edf_path.name.removesuffix("_eeg.edf")

        # Only use nights that have prepared EEG.
        if not (PROCESSED / f"{prefix}.npz").exists():
            report.append([prefix, "SKIPPED", "Missing NPZ"])
            continue

        try:
            subject, rows = process_recording(edf_path)
            rows = align_onset_rows(PROCESSED / f"{prefix}.npz", rows)

            group = subject_to_split[subject]
            results[group].extend(rows)
            successful += 1

            report.append([prefix, "OK", len(rows)])

            print(
                f"[{index}/{len(recordings)}] "
                f"{prefix}: {len(rows)} windows -> {group}",
                flush=True,
            )

        except Exception as error:
            report.append([prefix, "SKIPPED", str(error)])
            print(
                f"[{index}/{len(recordings)}] "
                f"SKIPPED {prefix}: {error}",
                flush=True,
            )

    for group, rows in results.items():
        path = OUTPUT / f"{group}_onset.csv"

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"{group}: {len(rows)} windows")

    with (OUTPUT / "report.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["recording", "status", "detail"])
        writer.writerows(report)

    print(
        f"\nCompleted: {successful}/{len(recordings)} recordings"
    )


if __name__ == "__main__":
    main()
