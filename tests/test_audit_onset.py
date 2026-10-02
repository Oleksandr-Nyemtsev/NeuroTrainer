import numpy as np
import pytest
from src.signal.sleep_onset import (make_onset_row, align_onset_rows, subject_split_lookup,
    ISRUC_ONSET_RULE, MUSE_ONSET_RULE, find_stable_n2)
from prepare_sleep_onset_all import process_recording


def make_npz(tmp_path, starts=(0., 30.)):
    path = tmp_path / "recording.npz"
    np.savez_compressed(path, X=np.zeros((len(starts), 2, 3000)), y=np.zeros(len(starts), dtype=int),
                        onsets=starts, window_start=starts, window_end=np.asarray(starts)+30.,
                        subject_id="s1", recording="r1")
    return path


def row(start, rule=ISRUC_ONSET_RULE):
    return make_onset_row("s1", "r1", start, start+30., 90., rule, stage=0)


def test_targets_use_window_end_and_preserve_definition():
    assert row(0.)["time_to_n2_seconds"] == 60.
    assert row(60.)["time_to_n2_seconds"] == 0.
    assert row(0.)["label_definition"] != row(0., MUSE_ONSET_RULE)["label_definition"]
    assert {"subject", "recording", "window_start", "window_end", "n2_onset"} <= row(0.).keys()


def test_alignment_is_by_time_not_row_order(tmp_path):
    rows = align_onset_rows(make_npz(tmp_path), [row(30.), row(0.)])
    assert [x["eeg_epoch_index"] for x in rows] == [1, 0]


@pytest.mark.parametrize("starts", [(0., 0.), (0., np.nan), (0., 60.)])
def test_reject_ambiguous_invalid_or_missing_windows(tmp_path, starts):
    with pytest.raises(ValueError):
        align_onset_rows(make_npz(tmp_path, starts), [row(0.), row(30.)])


def test_missing_metadata_and_wrong_subject_fail(tmp_path):
    path = make_npz(tmp_path)
    with pytest.raises(ValueError, match="identities"):
        align_onset_rows(path, [{**row(0.), "subject": "other"}])
    np.savez_compressed(path, X=np.zeros((1,2,3000)), y=[0])
    with pytest.raises(ValueError, match="regenerate"):
        align_onset_rows(path, [row(0.)])


def test_subjects_cannot_leak_between_splits():
    assert subject_split_lookup({"train":["a"], "val":["b"], "test":["c"]})["b"] == "val"
    with pytest.raises(ValueError):
        subject_split_lookup({"train":["a"], "val":["b"], "test":["a"]})


def test_isruc_event_timing(tmp_path):
    folder = tmp_path/"sub-test"/"eeg"
    folder.mkdir(parents=True)
    edf = folder/"sub-test_task-sleep_eeg.edf"
    events = edf.with_name("sub-test_task-sleep_events.tsv")
    events.write_text("onset\tduration\ttrial_type\n0\t30\tSleep stage W\n30\t30\tSleep stage N2\n60\t30\tSleep stage N2\n90\t30\tSleep stage N2\n")
    subject, rows = process_recording(edf)
    assert rows[0]["window_end"] == rows[0]["n2_onset"] == 30.
    assert rows[0]["time_to_n2_seconds"] == 0.
    events.write_text(events.read_text().replace("60\t30", "65\t30"))
    with pytest.raises(ValueError, match="Gap"):
        process_recording(edf)


def test_positive_stability_length_required():
    with pytest.raises(ValueError):
        find_stable_n2([2,2,2], required_epochs=0)


def test_muse_notebook_metadata_export_with_synthetic_windows(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    notebook = json.loads(Path("notebooks/01_muse_sleep_onset.ipynb").read_text(encoding="utf-8"))
    generation = next("".join(c.get("source", [])) for c in notebook["cells"]
                      if "window_metadata = pd.DataFrame" in "".join(c.get("source", [])))
    export = next("".join(c.get("source", [])) for c in notebook["cells"]
                  if "window_metadata.to_csv" in "".join(c.get("source", [])))
    raw = SimpleNamespace(info={"sfreq":100}, get_data=lambda start,stop: np.zeros((4,stop-start)))
    scope = {"raw":raw, "project_root":tmp_path, "subject":"sub-demo", "session":"ses-01", "n2_onset":60.}
    exec(compile(generation, "muse-window-cell", "exec"), scope)
    exec(compile(export, "muse-export-cell", "exec"), scope)
    metadata = pd.read_csv(scope["metadata_path"])
    assert metadata["window_end"].tolist() == [30.,60.]
    assert metadata["time_to_n2_seconds"].tolist() == [30.,0.]
    assert metadata["eeg_epoch_index"].tolist() == [0,1]
    assert set(metadata["label_definition"]) == {MUSE_ONSET_RULE}
    assert set(metadata["subject"]) == {"sub-demo"}
    with pytest.raises(FileExistsError):
        exec(compile(export, "muse-export-cell", "exec"), scope)
