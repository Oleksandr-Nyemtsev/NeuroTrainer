"""BOAS recording IDs (sub-*) are not participant IDs; split by participants.tsv pid."""
import csv
import random
from pathlib import Path

import numpy as np

GROUPS = ("train", "val", "test")


def build_participant_split(data_dir, participants_path, seed=42):
    mapping = {}
    with Path(participants_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"participant_id", "pid"}.issubset(reader.fieldnames or []):
            raise ValueError("participants.tsv must contain participant_id and pid")
        for row in reader:
            recording, pid = row["participant_id"].strip(), row["pid"].strip()
            if not recording or pid.lower() in ("", "n/a", "nan", "none"):
                raise ValueError("Missing recording ID or participant pid")
            if recording in mapping:
                raise ValueError(f"Duplicate recording ID: {recording}")
            mapping[recording] = pid

    records = {}
    for path in sorted(Path(data_dir).glob("sub-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            recording = str(archive["subject_id"].item())
        if recording != path.stem:
            raise ValueError(f"NPZ recording ID differs from filename: {path}")
        if recording not in mapping:
            raise ValueError(f"No participant pid for {recording}")
        records[recording] = {"pid": mapping[recording], "file": path.name}
    if not records:
        raise ValueError("No processed BOAS recordings")

    participants = sorted({r["pid"] for r in records.values()})
    random.Random(seed).shuffle(participants)
    n1, n2 = int(len(participants) * .70), int(len(participants) * .85)
    groups = dict(zip(GROUPS, (participants[:n1], participants[n1:n2], participants[n2:])))
    if any(not ids for ids in groups.values()):
        raise ValueError("Not enough participants for nonempty train/val/test")
    split = {
        "schema_version": 2,
        "split_unit": "participants.tsv:pid",
        "seed": seed,
        "participants": {g: sorted(groups[g]) for g in GROUPS},
        "recordings": {
            g: sorted(r for r, info in records.items() if info["pid"] in groups[g])
            for g in GROUPS
        },
        "recording_metadata": records,
    }
    validate_participant_split(split)
    return split


def validate_participant_split(split):
    if split.get("schema_version") != 2 or split.get("split_unit") != "participants.tsv:pid":
        raise ValueError("Legacy recording-level split is not a participant split")
    seen_pids, seen_records = set(), set()
    for group in GROUPS:
        pids, records = split["participants"][group], split["recordings"][group]
        if not pids or not records or len(pids) != len(set(pids)) or len(records) != len(set(records)):
            raise ValueError(f"Empty or duplicate split entries: {group}")
        if seen_pids.intersection(pids) or seen_records.intersection(records):
            raise ValueError("Participant or recording leakage")
        actual = {split["recording_metadata"][r]["pid"] for r in records}
        if actual != set(pids):
            raise ValueError(f"Recording/participant mismatch: {group}")
        seen_pids.update(pids)
        seen_records.update(records)
    if seen_records != set(split["recording_metadata"]):
        raise ValueError("Split does not cover all processed recordings")
