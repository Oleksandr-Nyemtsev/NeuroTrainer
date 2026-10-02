import time
from types import SimpleNamespace
import numpy as np
import pytest
import src.core.session as session_module
from src.signal.muse_quality import MuseSignalQuality
from src.signal.filters import bandpass_filter
from src.signal.metrics import relaxation_score
from src.core.brain_state import BrainState
from src.core.brain_state_estimator import BrainStateEstimator
from src.core.controller import choose_audio_action


def make_session(monkeypatch, eeg):
    monkeypatch.setattr(session_module, "BoardShim", SimpleNamespace(get_eeg_channels=lambda _: [0, 1, 2, 3], get_timestamp_channel=lambda _: 4))
    actions = []
    audio = SimpleNamespace(apply_action=actions.append, target_master_volume=0.1)
    def data():
        ts = time.time() - (eeg.shape[1]-1-np.arange(eeg.shape[1]))/256
        return np.vstack([eeg, ts])
    muse = SimpleNamespace(get_data=data)
    return session_module.NeuroSession(muse, audio), actions


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("whole_channel", [False, True])
def test_invalid_eeg_never_reaches_filter(monkeypatch, bad, whole_channel):
    eeg = np.tile(20*np.sin(2*np.pi*10*np.arange(512)/256), (4, 1))
    eeg[0, :] = bad if whole_channel else eeg[0, :]
    eeg[0, 20] = bad
    quality = MuseSignalQuality().analyze(eeg)
    assert all(set(c) == set(quality["channels"][0]) for c in quality["channels"])
    assert quality["channels"][0]["quality"] == 0
    session, actions = make_session(monkeypatch, eeg)
    def unexpected(*a, **k):
        pytest.fail("Invalid EEG reached signal filtering")
    monkeypatch.setattr(session_module, "process_eeg", unexpected)
    assert session.run_step()["ready"] is False
    assert actions[-1] == {"master_volume": 0.0}


def test_loss_and_recovery(monkeypatch):
    eeg = np.tile(20*np.sin(2*np.pi*10*np.arange(512)/256), (4, 1))
    session, actions = make_session(monkeypatch, eeg)
    get_valid_data = session.muse.get_data
    session.muse.get_data = lambda: None
    assert not session.run_step()["ready"]
    session.muse.get_data = get_valid_data
    assert session.run_step()["ready"]
    assert actions[-1]["master_volume"] == 0.1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_state_score_and_controller(bad):
    with pytest.raises(ValueError):
        BrainState(alpha_phase=bad)
    with pytest.raises(ValueError):
        relaxation_score({"Theta": bad, "Alpha": 10., "Beta": 2.})
    with pytest.raises(ValueError):
        bandpass_filter(np.full(512, bad))
    assert choose_audio_action(bad) == {"master_volume": 0.0}


def test_estimator_rejects_invalid_input_before_predict():
    def unexpected(eeg):
        pytest.fail("Invalid EEG reached model")
    estimator = BrainStateEstimator(SimpleNamespace(predict=unexpected))
    state = estimator.estimate(np.zeros((2, 3000)), np.full((4, 512), np.nan))
    assert state.signal_quality == 0
    assert state.is_valid()


def test_valid_controller_behavior_preserved():
    assert [choose_audio_action(s)["beat_frequency"] for s in [1., 2., 4.]] == [6., 8., 10.]
