"""Causal Muse Sleep-Onset preprocessing. No filters, resampling or training."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import mne
import numpy as np

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
WINDOW_SECONDS = 5
STEP_SECONDS = 5
SEED = 42
LABEL = "clip(n2_onset - window_end, 0, 600); seconds; window_end <= n2_onset"
FIELDS = [
    "window_id", "subject", "session", "recording_id", "window_index",
    "window_start", "window_end", "n2_onset", "target_seconds_to_n2",
    "sampling_frequency", "n_channels", "n_samples", "label_definition",
    "split", "eeg_file", "start_sample", "end_sample",
]


def target_seconds(n2_onset, window_end):
    if not np.isfinite([n2_onset, window_end]).all():
        raise ValueError("Nonfinite target timing")
    return float(np.clip(n2_onset - window_end, 0, 600))


def samples_per_window(sfreq):
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Invalid sfreq")
    samples = WINDOW_SECONDS * sfreq
    if not np.isclose(samples, round(samples), rtol=0, atol=1e-8):
        raise ValueError("5 seconds is not an integral number of samples; report a timing strategy first")
    return int(round(samples))


def validate_channels(names):
    if any(list(names).count(name) != 1 for name in CHANNELS):
        raise ValueError(f"Missing or duplicate Muse channel: {names}")


def read_n2(events_path, sfreq, n_times):
    with events_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t")
                if row.get("trial_type") == "n2_onset"]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one n2_onset, found {len(rows)}")
    onset = float(rows[0]["onset"])
    if not np.isfinite(onset) or not 0 <= onset < n_times / sfreq:
        raise ValueError(f"Invalid n2_onset: {onset}")
    if "sample" in rows[0] and rows[0]["sample"] not in ("", "n/a"):
        sample = float(rows[0]["sample"])
        if not np.isfinite(sample) or sample != int(sample) or abs(sample / sfreq - onset) > 1 / sfreq:
            raise ValueError("n2_onset sample/onset mismatch")
    return onset


def split_subjects(subjects, seed=SEED):
    # Same algorithm and rounding as src/data/sleep_dataset.py, without importing torch.
    subjects = sorted(set(subjects))
    np.random.default_rng(seed).shuffle(subjects)
    train_end = int(len(subjects) * 0.70)
    val_end = int(len(subjects) * (0.70 + 0.15))
    return {"train": sorted(subjects[:train_end]),
            "val": sorted(subjects[train_end:val_end]),
            "test": sorted(subjects[val_end:])}


def split_lookup(groups):
    lookup = {}
    for group, subjects in groups.items():
        for subject in subjects:
            if subject in lookup:
                raise ValueError(f"Subject leakage: {subject}")
            lookup[subject] = group
    return lookup


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def inspect_recording(header, root):
    relative = header.relative_to(root)
    subject = next(part for part in relative.parts if part.startswith("sub-"))
    session = next(part for part in relative.parts if part.startswith("ses-"))
    if not header.name.endswith("_eeg.vhdr"):
        raise ValueError("Expected BIDS *_eeg.vhdr")
    recording_id = header.name.removesuffix("_eeg.vhdr")
    events = header.with_name(recording_id + "_events.tsv")
    raw = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        validate_channels(raw.ch_names)
        samples_per_window(sfreq)
        onset = read_n2(events, sfreq, raw.n_times)
        return {
            "subject": subject, "session": session, "recording_id": recording_id,
            "header": relative.as_posix(), "events": events.relative_to(root).as_posix(),
            "sampling_frequency": sfreq, "source_channels": raw.ch_names,
            "source_n_samples": int(raw.n_times), "n2_onset": onset,
            "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
            "events_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
            "eeg_size_bytes": header.with_suffix(".eeg").stat().st_size,
            "eeg_mtime_ns": header.with_suffix(".eeg").stat().st_mtime_ns,
        }
    finally:
        raw.close()


def check_sfreq_distribution(records, allow_mixed=False):
    distribution = dict(sorted(Counter(str(r["sampling_frequency"]) for r in records).items()))
    if len(distribution) > 1 and not allow_mixed:
        raise ValueError("Mixed sfreq; see inventory.json. Proposed strategy: preserve per-recording "
                         "native sfreq and shapes with --allow-mixed-sfreq after report review. "
                         "No automatic resampling.")
    return distribution


def make_windows(raw, onset):
    validate_channels(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    width = samples_per_window(sfreq)
    if not np.isfinite(onset) or not 0 <= onset < raw.n_times / sfreq:
        raise ValueError("Invalid n2_onset")
    count = int(np.floor(onset / WINDOW_SECONDS))
    if count == 0:
        raise ValueError("No complete pre-N2 window")
    # Read only [0, last window_end), never EEG at or after the N2 boundary.
    stop = count * width
    if stop / sfreq > onset or stop > raw.n_times:
        raise ValueError("Window exceeds N2 or recording boundary")
    raw.pick(list(CHANNELS))
    if tuple(raw.ch_names) != CHANNELS:
        raise ValueError("Channel order mismatch")
    data = raw.get_data(start=0, stop=stop)
    if data.shape != (4, stop) or not np.isfinite(data).all():
        raise ValueError("Invalid EEG shape or nonfinite pre-N2 samples")
    windows = data.reshape(4, count, width).transpose(1, 0, 2).copy()
    if windows.shape != (count, 4, width):
        raise ValueError("Invalid window shape")
    return windows


def process_recording(record, root, output, lookup):
    raw = mne.io.read_raw_brainvision(root / record["header"], preload=False, verbose="ERROR")
    try:
        if float(raw.info["sfreq"]) != record["sampling_frequency"]:
            raise ValueError("sfreq changed since inventory")
        windows = make_windows(raw, record["n2_onset"])
    finally:
        raw.close()
    recording_id = record["recording_id"]
    ids = np.array([f"{recording_id}:window-{i:06d}" for i in range(len(windows))])
    eeg_file = f"eeg/{recording_id}.npz"
    # Exclusive creation prevents replacing an existing archive.
    with (output / eeg_file).open("xb") as handle:
        np.savez_compressed(handle, X=windows, window_ids=ids, channels=np.array(CHANNELS),
                            sampling_frequency=record["sampling_frequency"], units="V")
    rows = []
    for i, window_id in enumerate(ids):
        start = i * windows.shape[2]
        end = start + windows.shape[2]
        sfreq = record["sampling_frequency"]
        rows.append(dict(zip(FIELDS, [
            str(window_id), record["subject"], record["session"], recording_id, i,
            start / sfreq, end / sfreq, record["n2_onset"],
            target_seconds(record["n2_onset"], end / sfreq), sfreq, 4, windows.shape[2],
            LABEL, lookup[record["subject"]], eeg_file, start, end,
        ])))
    return rows


def verify_alignment(output, rows, lookup):
    """Reopen saved archives and verify stable IDs, timing, shapes and split."""
    grouped = {}
    seen = set()
    for row in rows:
        if row["window_id"] in seen or lookup[row["subject"]] != row["split"]:
            raise ValueError("Duplicate window ID or subject split mismatch")
        seen.add(row["window_id"])
        grouped.setdefault(row["eeg_file"], []).append(row)
    for filename, entries in grouped.items():
        with np.load(output / filename, allow_pickle=False) as archive:
            x = archive["X"]
            if tuple(archive["channels"]) != CHANNELS or not np.isfinite(x).all():
                raise ValueError("Archive channel order or finite check failed")
            if len(x) != len(entries) or x.shape[1:] != (4, int(entries[0]["n_samples"])):
                raise ValueError("Archive shape mismatch")
            indices = set()
            for row in entries:
                index = int(row["window_index"])
                indices.add(index)
                sfreq = float(row["sampling_frequency"])
                if (str(archive["window_ids"][index]) != row["window_id"]
                        or float(archive["sampling_frequency"]) != sfreq
                        or int(row["n_channels"]) != 4
                        or int(row["end_sample"]) - int(row["start_sample"]) != x.shape[2]
                        or int(row["start_sample"]) != index * x.shape[2]
                        or float(row["window_start"]) != int(row["start_sample"]) / sfreq
                        or float(row["window_end"]) != int(row["end_sample"]) / sfreq
                        or float(row["window_end"]) > float(row["n2_onset"])
                        or float(row["target_seconds_to_n2"]) != target_seconds(
                            float(row["n2_onset"]), float(row["window_end"]))
                        or row["label_definition"] != LABEL):
                    raise ValueError("Metadata/EEG alignment mismatch")
            if indices != set(range(len(x))):
                raise ValueError("Missing or repeated window index")
    return len(seen)


def run(args):
    root = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source dataset")
    headers = sorted(root.rglob("*.vhdr"))
    if not headers:
        raise ValueError("No BrainVision headers found")
    # An existing output directory is always refused, even for a scan or dry run.
    output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    for header in headers:
        try:
            records.append(inspect_recording(header, root))
        except Exception as error:
            failures.append({"header": header.relative_to(root).as_posix(),
                             "phase": "inventory", "error": str(error)})
    subjects = sorted({part for header in headers for part in header.relative_to(root).parent.parts
                       if part.startswith("sub-")})
    groups = split_subjects(subjects, args.seed)
    lookup = split_lookup(groups)
    distribution = dict(sorted(Counter(str(r["sampling_frequency"]) for r in records).items()))
    inventory = {"data_dir": str(root), "discovered_recordings": len(headers),
                 "subjects": subjects, "sfreq_distribution": distribution,
                 "records": records, "failures": failures}
    write_json(output / "inventory.json", inventory)
    write_json(output / "split_manifest.json", {
        "seed": args.seed, "ratios": [0.70, 0.15, 0.15], "splits": groups,
        "algorithm": "sorted unique subjects; numpy.default_rng(seed).shuffle; floor cutoffs",
        "scope": "all discovered subjects, including dry-run and failed recordings",
    })
    print(f"Inventory: {len(headers)} recordings, {len(subjects)} subjects; sfreq={distribution}", flush=True)
    check_sfreq_distribution(records, args.allow_mixed_sfreq)
    if args.scan_only:
        return 1 if failures else 0
    selected = [r for r in records if not args.subject or r["subject"] in args.subject]
    if args.subject and set(args.subject) - set(subjects):
        raise ValueError("Unknown requested subject")
    if args.max_recordings:
        selected = selected[:args.max_recordings]
    (output / "eeg").mkdir()
    rows, successful = [], []
    for i, record in enumerate(selected, 1):
        try:
            new_rows = process_recording(record, root, output, lookup)
            rows.extend(new_rows)
            successful.append(record["recording_id"])
            print(f"[{i}/{len(selected)}] {record['recording_id']}: {len(new_rows)} windows", flush=True)
        except Exception as error:
            failures.append({"header": record["header"], "phase": "processing", "error": str(error)})
            print(f"FAILED {record['recording_id']}: {error}", flush=True)
    write_csv(output / "windows.csv", rows)
    for group in groups:
        write_csv(output / f"{group}.csv", [r for r in rows if r["split"] == group])
    # Verify serialized metadata, not only the in-memory rows.
    with (output / "windows.csv").open(encoding="utf-8", newline="") as handle:
        checked = verify_alignment(output, list(csv.DictReader(handle)), lookup)
    targets = np.array([r["target_seconds_to_n2"] for r in rows])
    from muse_sleep_onset_metrics import binned_mae_seconds
    bins = binned_mae_seconds(targets, targets) if len(targets) else {}
    report = {
        "status": "complete" if not failures and rows else "incomplete",
        "discovered_subjects": len(subjects), "discovered_recordings": len(headers),
        "selected_recordings": len(selected), "successful_recordings": len(successful),
        "processed_subjects": len({r["subject"] for r in rows}), "windows": len(rows),
        "sfreq_distribution": distribution, "channel_order": list(CHANNELS),
        "channel_validation_passed": len(records), "finite_and_alignment_windows_verified": checked,
        "failed_recordings": failures, "unselected_recordings": len(records) - len(selected),
        "target_distribution": {"min": float(targets.min()) if len(targets) else None,
                                "max": float(targets.max()) if len(targets) else None,
                                "mean": float(targets.mean()) if len(targets) else None,
                                "quantiles_0_25_50_75_100": np.quantile(targets, [0,.25,.5,.75,1]).tolist() if len(targets) else [],
                                "clipped_at_600": int((targets == 600).sum()),
                                "zero": int((targets == 0).sum()),
                                "bins": {key: value["count"] for key, value in bins.items()}},
        "split_subject_counts": {g: len(s) for g, s in groups.items()},
        "processed_split_subject_counts": {g: len({r["subject"] for r in rows if r["split"] == g}) for g in groups},
        "split_window_counts": dict(Counter(r["split"] for r in rows)),
        "subject_leakage": 0, "window_seconds": WINDOW_SECONDS, "step_seconds": STEP_SECONDS,
        "label_definition": LABEL, "units": "V", "dtype": "float64",
        "preprocessing": "MNE BrainVision scaling only; native sfreq; no filtering/normalization/resampling",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "mne": mne.__version__},
        "command": sys.argv, "data_dir": str(root), "output_dir": str(output),
        "disk_bytes_before_report": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "risks": ["Supplied recording split overlaps subjects; replaced with subject-only split.",
                  "Metadata contains N2 and recording identifiers: labels/provenance only, never model features.",
                  "No artifact correction; acquisition filtering and original resampling provenance unknown.",
                  "Only complete pre-N2 windows; final partial window discarded; no post-N2 zero-label windows.",
                  "Source EEG fingerprints are size/mtime, not content hashes; header/events use SHA256."],
    }
    write_json(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "windows", "successful_recordings", "split_window_counts")}), flush=True)
    return 0 if report["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/muse_sleep_onset"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths refused")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--subject", action="append", help="BIDS subject e.g. sub-001; repeatable")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--allow-mixed-sfreq", action="store_true", help="Explicitly preserve native per-recording rates")
    args = parser.parse_args()
    if args.max_recordings is not None and args.max_recordings <= 0:
        parser.error("--max-recordings must be positive")
    try:
        return run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
