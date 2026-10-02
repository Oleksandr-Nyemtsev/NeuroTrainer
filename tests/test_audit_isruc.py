from types import SimpleNamespace
import numpy as np
import prepare_isruc


def test_prepare_recording_npz_roundtrip(tmp_path, monkeypatch):
    eeg_dir = tmp_path / "sub-test" / "eeg"
    eeg_dir.mkdir(parents=True)
    edf = eeg_dir / "sub-test_task-sleep_eeg.edf"
    edf.with_name("sub-test_task-sleep_events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        "0\t30\tSleep stage W\n30\t30\tSleep stage N2\n"
    )
    raw = SimpleNamespace(info={"sfreq": 100}, close=lambda: None)
    monkeypatch.setattr(prepare_isruc.mne.io, "read_raw_edf", lambda *a, **k: raw)
    eeg = np.random.default_rng(42).normal(size=(2, 6000))
    monkeypatch.setattr(prepare_isruc, "select_eeg", lambda raw: (eeg, "M2", ["C3-M2", "O1-M2"]))
    monkeypatch.setattr(prepare_isruc, "OUTPUT", tmp_path / "processed")
    result = prepare_isruc.prepare_recording(edf)
    assert result["status"] == "OK"
    with np.load(prepare_isruc.OUTPUT / "sub-test_task-sleep.npz", allow_pickle=False) as data:
        assert data["X"].shape == (2, 2, 3000)
        assert np.isfinite(data["X"]).all()
        np.testing.assert_array_equal(data["y"], [0, 2])
        np.testing.assert_array_equal(data["onsets"], [0., 30.])
        np.testing.assert_array_equal(data["source_channels"], ["C3-M2", "O1-M2"])
        np.testing.assert_array_equal(data["window_start"], [0., 30.])
        np.testing.assert_array_equal(data["window_end"], [30., 60.])
        assert data["recording"].item() == "sub-test_task-sleep"
        assert data["subject_id"].item() == "sub-test"
        assert data["sampling_frequency"].item() == 100
