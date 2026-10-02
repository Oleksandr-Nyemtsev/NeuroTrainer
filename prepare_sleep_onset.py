from src.signal.sleep_onset import (
    find_stable_n2, make_onset_row, align_onset_rows, ONSET_FIELDS, ISRUC_ONSET_RULE,
)
from pathlib import Path
import csv


DATA_DIR = Path("data/external/isruc/nm000111")
OUTPUT_DIR = Path("data/processed/sleep_onset")

PROCESSED = Path("data/processed/isruc_v2")
SUBJECT = "sub-III001"

STAGES = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}


def prepare_sleep_onset(subject):
    events_file = (
        DATA_DIR
        / subject
        / "eeg"
        / f"{subject}_task-sleep_events.tsv"
    )

    # Read TSV using Python's built-in csv module.
    with events_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        events = list(reader)

    # Select recognized 30-second sleep-stage epochs.
    valid = []

    for event in events:
        stage = event["trial_type"]

        if stage not in STAGES:
            continue

        onset = float(event["onset"])
        duration = float(event["duration"])

        if abs(duration - 30.0) > 0.01:
            continue

        valid.append({
            "onset": onset,
            "stage": STAGES[stage],
        })

    valid.sort(key=lambda event: event["onset"])

    if not valid:
        raise ValueError("No valid sleep-stage epochs")

    # Find stable N2
    stages = [event["stage"] for event in valid]

    stable_n2_index = find_stable_n2(
        stages,
        required_epochs=3,
    )

    if stable_n2_index is None:
        raise ValueError("No stable N2 sequence found")

    stable_times = [event["onset"] for event in valid[stable_n2_index:stable_n2_index + 3]]
    if any(abs(b - a - 30.0) > 0.01 for a, b in zip(stable_times, stable_times[1:])):
        raise ValueError("Gap inside stable N2")
    stable_n2_time = valid[stable_n2_index]["onset"]
    before_n2 = valid[:stable_n2_index]
    first_n2_time = stable_n2_time

    if not before_n2:
        raise ValueError("Recording starts with stable N2")



    if not before_n2:
        raise ValueError("Recording starts with N2")

    # Check for missing or duplicate 30-second epochs.
    timeline = [
        event["onset"]
        for event in before_n2
    ] + [first_n2_time]

    for previous, current in zip(
        timeline,
        timeline[1:],
    ):
        if abs((current - previous) - 30.0) > 0.01:
            raise ValueError(
                "Gap or duplicate in pre-N2 timeline"
            )

    recording = events_file.name.removesuffix("_events.tsv")
    rows = [make_onset_row(subject, recording, event["onset"], event["onset"] + 30.0,
                           stable_n2_time, ISRUC_ONSET_RULE, stage=event["stage"])
            for event in before_n2]
    rows = align_onset_rows(PROCESSED / f"{recording}.npz", rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR / f"{subject}_onset.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ONSET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Subject:", subject)
    print("First N2:", first_n2_time, "seconds")
    print("Training windows:", len(before_n2))
    print("Saved:", output_file)

    print("\nFirst five windows:")

    for event in before_n2[:5]:
        print(
            "Time:", event["onset"],
            "Stage:", event["stage"],
            "Seconds to N2:",
            first_n2_time - (event["onset"] + 30.0),
        )


if __name__ == "__main__":
    prepare_sleep_onset(SUBJECT)
