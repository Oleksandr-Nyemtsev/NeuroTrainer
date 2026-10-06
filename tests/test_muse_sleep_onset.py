import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np
import pytest

import prepare_muse_sleep_onset_all as pipeline
from muse_sleep_onset_metrics import mae_seconds, binned_mae_seconds


def raw_array(sfreq=10, names=("AF8", "TP10", "TP9", "AF7")):
    data = np.array([np.full(200, pipeline.CHANNELS.index(name) + 1) for name in names], dtype=float)
    return mne.io.RawArray(data, mne.create_info(list(names), sfreq, "eeg"), verbose="ERROR")


@pytest.mark.parametrize("onset,end,expected", [(100, 30, 70), (100, 100, 0),
                                               (100, 110, 0), (1000, 5, 600), (600, 0, 600)])
def test_target(onset, end, expected):
    assert pipeline.target_seconds(onset, end) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_target(value):
    with pytest.raises(ValueError):
        pipeline.target_seconds(value, 5)


def test_shape_order_and_no_future():
    raw = raw_array()
    raw._data[:, 100:] = np.nan  # Future samples must never reach finite check or output.
    windows = pipeline.make_windows(raw, 10)
    assert windows.shape == (2, 4, 50)
    np.testing.assert_array_equal(windows[0, :, 0], [1, 2, 3, 4])
    assert np.isfinite(windows).all()


def test_read_stops_at_last_complete_window(monkeypatch):
    raw = raw_array()
    original = raw.get_data
    calls = []
    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(raw, "get_data", tracked)
    assert pipeline.make_windows(raw, 12.7).shape == (2, 4, 50)
    assert calls == [{"start": 0, "stop": 100}]


def test_nonfinite_past_rejected():
    raw = raw_array()
    raw._data[0, 10] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        pipeline.make_windows(raw, 10)


@pytest.mark.parametrize("names", [("TP9", "AF7", "AF8"), ("TP9", "AF7", "AF8", "AF8"),
                                  ("TP9", "AF7", "AF8", "tp10")])
def test_invalid_channels(names):
    with pytest.raises(ValueError, match="channel"):
        pipeline.validate_channels(names)


@pytest.mark.parametrize("sfreq", [0, -1, np.nan, np.inf, 10.1])
def test_invalid_frequency(sfreq):
    with pytest.raises(ValueError):
        pipeline.samples_per_window(sfreq)


def test_native_frequency_and_mixed_policy():
    assert pipeline.make_windows(raw_array(20), 5).shape == (1, 4, 100)
    records = [{"sampling_frequency": 10.}, {"sampling_frequency": 20.}]
    with pytest.raises(ValueError, match="Mixed sfreq"):
        pipeline.check_sfreq_distribution(records)
    assert pipeline.check_sfreq_distribution(records, True) == {"10.0": 1, "20.0": 1}


def test_split_deterministic_and_leak_free():
    subjects = [f"sub-{i:03}" for i in range(1, 204)]
    groups = pipeline.split_subjects(subjects)
    assert groups == pipeline.split_subjects(list(reversed(subjects)) + subjects[:4])
    assert groups != pipeline.split_subjects(subjects, 43)
    assert [len(groups[g]) for g in ("train", "val", "test")] == [142, 30, 31]
    assert len(pipeline.split_lookup(groups)) == 203
    with pytest.raises(ValueError, match="leakage"):
        pipeline.split_lookup({"train": ["sub-001"], "test": ["sub-001"]})


def event_file(path, rows):
    path.write_text("onset\ttrial_type\tsample\n" + rows, encoding="utf-8")


@pytest.mark.parametrize("rows", ["", "nan\tn2_onset\t0\n", "inf\tn2_onset\t0\n",
    "-1\tn2_onset\t0\n", "20\tn2_onset\t200\n", "5\tother\t50\n",
    "5\tn2_onset\t50\n6\tn2_onset\t60\n", "5\tn2_onset\t90\n",
    "abc\tn2_onset\t50\n", "5\tn2_onset\t50.5\n"])
def test_invalid_n2(tmp_path, rows):
    path = tmp_path / "events.tsv"
    event_file(path, rows)
    with pytest.raises(ValueError):
        pipeline.read_n2(path, 10, 200)


def test_missing_and_valid_n2(tmp_path):
    path = tmp_path / "events.tsv"
    with pytest.raises(FileNotFoundError):
        pipeline.read_n2(path, 10, 200)
    event_file(path, "5\tn2_onset\t50\n")
    assert pipeline.read_n2(path, 10, 200) == 5


@pytest.mark.parametrize("onset", [0, 4.9, -1, np.nan, 20])
def test_no_valid_windows(onset):
    with pytest.raises(ValueError):
        pipeline.make_windows(raw_array(), onset)


def brainvision_fixture(root, subject, sfreq=10):
    folder = root / subject / "ses-001" / "eeg"
    folder.mkdir(parents=True)
    prefix = f"{subject}_ses-001_task-sleeponset"
    header = folder / f"{prefix}_eeg.vhdr"
    # Minimal valid BrainVision triplet with distinguishable channel values.
    names = ["AF8", "TP10", "TP9", "AF7"]
    data = np.array([[3, 4, 1, 2]] * int(20 * sfreq), dtype="<f4")
    data.tofile(header.with_suffix(".eeg"))
    header.with_suffix(".vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\n"
        f"DataFile={header.stem}.eeg\n[Marker Infos]\n", encoding="utf-8")
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
        f"DataFile={header.stem}.eeg\nMarkerFile={header.stem}.vmrk\n"
        f"DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=4\nSamplingInterval={1e6/sfreq}\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n" +
        "".join(f"Ch{i}={name},,1,µV\n" for i, name in enumerate(names, 1)), encoding="utf-8")
    event_file(folder / f"{prefix}_events.tsv", f"10\tn2_onset\t{int(10*sfreq)}\n")
    return header


def arguments(root, output, **kwargs):
    options = dict(data_dir=root, output_dir=output, seed=42, scan_only=False,
                   subject=None, max_recordings=None, allow_mixed_sfreq=False)
    options.update(kwargs)
    return argparse.Namespace(**options)


def test_brainvision_end_to_end_alignment_and_no_overwrite(tmp_path):
    root, output = tmp_path / "source", tmp_path / "output"
    for i in range(1, 8):
        brainvision_fixture(root, f"sub-{i:03}")
    assert pipeline.run(arguments(root, output)) == 0
    with (output / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    groups = json.loads((output / "split_manifest.json").read_text())["splits"]
    assert sum(map(len, groups.values())) == 7
    assert set(pipeline.split_lookup(groups)) == {f"sub-{i:03}" for i in range(1, 8)}
    lookup = pipeline.split_lookup(groups)
    assert pipeline.verify_alignment(output, rows[::-1], lookup) == 14
    with np.load(output / rows[0]["eeg_file"]) as archive:
        np.testing.assert_allclose(archive["X"][0, :, 0], np.arange(1, 5) * 1e-6)
    for split in groups:
        with (output / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            split_rows = list(csv.DictReader(handle))
        assert split_rows == [row for row in rows if row["split"] == split]
    rows[0]["window_id"] = "wrong"
    with pytest.raises(ValueError, match="alignment"):
        pipeline.verify_alignment(output, rows, lookup)
    with pytest.raises(FileExistsError):
        pipeline.run(arguments(root, output))


def test_mixed_rates_report_before_processing(tmp_path):
    root, output = tmp_path / "source", tmp_path / "output"
    brainvision_fixture(root, "sub-001", 10)
    brainvision_fixture(root, "sub-002", 20)
    with pytest.raises(ValueError, match="Mixed sfreq"):
        pipeline.run(arguments(root, output))
    assert json.loads((output / "inventory.json").read_text())["sfreq_distribution"] == {"10.0": 1, "20.0": 1}
    assert not (output / "eeg").exists()
    assert pipeline.run(arguments(root, tmp_path / "explicit", allow_mixed_sfreq=True)) == 0


def test_failed_recording_is_reported(tmp_path):
    root, output = tmp_path / "source", tmp_path / "output"
    header = brainvision_fixture(root, "sub-001")
    event_file(header.with_name(header.name.replace("_eeg.vhdr", "_events.tsv")), "")
    assert pipeline.run(arguments(root, output)) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["windows"] == 0
    assert "n2_onset" in report["failed_recordings"][0]["error"]


def test_metrics_boundaries_and_empty_bins():
    targets = [0, 39, 40, 89, 90, 299, 300, 600]
    predictions = np.array(targets) + np.arange(1, 9)
    assert mae_seconds(targets, predictions) == 4.5
    bins = binned_mae_seconds(targets, predictions)
    assert [v["count"] for v in bins.values()] == [2, 2, 2, 2]
    assert [v["mae_seconds"] for v in bins.values()] == [1.5, 3.5, 5.5, 7.5]
    assert binned_mae_seconds([0], [3])["40-90"] == {"count": 0, "mae_seconds": None}


@pytest.mark.parametrize("true,pred", [([], []), ([1], [1, 2]), ([[1]], [[1]]),
                                    ([np.nan], [1]), ([1], [np.inf])])
def test_invalid_metrics(true, pred):
    with pytest.raises(ValueError):
        mae_seconds(true, pred)


def test_out_of_range_binned_target():
    with pytest.raises(ValueError):
        binned_mae_seconds([601], [600])
