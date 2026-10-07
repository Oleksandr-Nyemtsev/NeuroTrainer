import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.boas_split import build_participant_split, validate_participant_split


def fixture(tmp_path, n=24):
    data = tmp_path / "processed"
    data.mkdir()
    rows = ["participant_id\tpid"]
    for i in range(n):
        # Two distinct recording IDs per actual person, as in BOAS.
        for night in range(2):
            name = f"sub-{2*i+night+1}"
            rows.append(f"{name}\t{i+1}")
            np.savez(data / f"{name}.npz", subject_id=name)
    table = tmp_path / "participants.tsv"
    table.write_text("\n".join(rows)+"\n", encoding="utf-8")
    return data, table


def test_same_person_nights_never_cross_groups(tmp_path):
    data, table = fixture(tmp_path)
    split = build_participant_split(data, table)
    assignments = {}
    for group, records in split["recordings"].items():
        for rec in records:
            pid = split["recording_metadata"][rec]["pid"]
            assert assignments.setdefault(pid, group) == group
    assert len(assignments) == 24
    assert sum(map(len, split["recordings"].values())) == 48
    assert split == build_participant_split(data, table)
    assert split != build_participant_split(data, table, seed=43)


def test_leakage_and_legacy_rejected(tmp_path):
    data, table = fixture(tmp_path)
    split = build_participant_split(data, table)
    bad = copy.deepcopy(split)
    bad["participants"]["test"].append(bad["participants"]["train"][0])
    with pytest.raises(ValueError, match="leakage"):
        validate_participant_split(bad)
    with pytest.raises(ValueError, match="Legacy"):
        validate_participant_split({"train": ["sub-1"], "seed": 42})


@pytest.mark.parametrize("problem", ["missing", "duplicate", "empty_pid", "npz_mismatch"])
def test_identity_errors_fail_closed(tmp_path, problem):
    data, table = fixture(tmp_path)
    text = table.read_text()
    if problem == "missing": text = text.replace("sub-1\t1\n", "")
    elif problem == "duplicate": text += "sub-1\t99\n"
    elif problem == "empty_pid": text = text.replace("sub-1\t1\n", "sub-1\tn/a\n")
    else: np.savez(data / "sub-1.npz", subject_id="sub-999")
    table.write_text(text)
    with pytest.raises(ValueError):
        build_participant_split(data, table)


def test_prepare_only_no_training_and_no_overwrite(tmp_path):
    data, table = fixture(tmp_path)
    checkpoint = tmp_path / "isruc.pt"
    checkpoint.write_bytes(b"not loaded in prepare-only mode")
    out = tmp_path / "new_run"
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-B", str(root / "train_boas_hybrid.py"),
           "--prepare-only", "--data-dir", str(data), "--participants", str(table),
           "--pretrained", str(checkpoint), "--output-dir", str(out)]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    manifest = out / "participant_split.json"
    original = manifest.read_bytes()
    assert not (out / "hybrid_boas_best.pt").exists()
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert manifest.read_bytes() == original
    (out / "history.csv").write_text("existing results")
    refused = subprocess.run(cmd, capture_output=True, text=True)
    assert refused.returncode != 0
    assert (out / "history.csv").read_text() == "existing results"


def test_import_does_not_start_training():
    import train_boas_hybrid
    assert callable(train_boas_hybrid.main)
