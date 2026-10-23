import math
from types import SimpleNamespace
import numpy as np
import pytest
from src.core.audio_experiment import AudioExperiment
import src.core.session as session_module


def test_silence_between_all_three_candidates():
    policy = AudioExperiment(42)
    seen = []
    for t in range(0, 362, 2):
        action = policy.choose(1, {"ready": True}, t)
        assert 0 <= action["master_volume"] <= .03
        assert 6 <= action["beat_frequency"] <= 10
        if t % 60 == 0:
            seen.append(policy.candidate)
    assert seen[::2] == ["silence"] * 4
    assert set(seen[1::2]) == {"beat_6", "beat_8", "beat_10"}


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), 59, 70])
def test_invalid_or_discontinuous_time_restarts_silence(bad_time):
    policy = AudioExperiment()
    for t in range(0, 62, 2):
        policy.choose(1, {"ready": True}, t)
    assert policy.candidate != "silence"
    assert policy.choose(1, {"ready": True}, bad_time)["master_volume"] == 0


def test_sleep_predictions_do_not_change_schedule():
    a, b = AudioExperiment(), AudioExperiment()
    for t in range(0, 400, 2):
        assert a.choose(0, {"ready": True, "stage": "REM"}, t) == b.choose(99, {"ready": True, "stage": "Wake"}, t)


def test_session_without_sleep_model_and_signal_loss(monkeypatch):
    now = [1000.]
    monkeypatch.setattr(session_module.time, "time", lambda: now[0])
    monkeypatch.setattr(session_module, "BoardShim", SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3], get_timestamp_channel=lambda _: 4))
    actions = []
    def data():
        signal = np.tile(20*np.sin(2*np.pi*10*np.arange(512)/256), (4,1))
        return np.vstack([signal, now[0]-(511-np.arange(512))/256])
    muse = SimpleNamespace(get_data=data)
    policy = AudioExperiment()
    session = session_module.NeuroSession(muse, SimpleNamespace(apply_action=actions.append), audio_policy=policy)
    for i in range(31):
        now[0] = 1000 + 2*i
        result = session.run_step()
        assert result["ready"]
        assert "sleep" not in result
    assert actions[-1]["master_volume"] == .03
    muse.get_data = lambda: None
    assert not session.run_step()["ready"]
    assert actions[-1]["master_volume"] == 0
    muse.get_data = data
    now[0] += 2
    assert session.run_step()["action"]["master_volume"] == 0


def test_variable_ble_read_sizes_do_not_restart_silent_reference(monkeypatch):
    now = [1000.]
    monkeypatch.setattr(session_module.time, "time", lambda: now[0])
    monkeypatch.setattr(session_module, "BoardShim", SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3], get_timestamp_channel=lambda _: 4))
    actions = []
    muse = SimpleNamespace()
    policy = AudioExperiment()
    session = session_module.NeuroSession(muse, SimpleNamespace(apply_action=actions.append), audio_policy=policy)
    offset = 0
    for n in [512] + [504, 516, 516] * 12:
        indices = np.arange(offset, offset+n)
        ts = 1000 + indices/256
        eeg = np.tile(20*np.sin(2*np.pi*10*indices/256), (4, 1))
        chunk = np.vstack([eeg, ts])
        muse.get_data = lambda: chunk
        now[0] = ts[-1] + .01
        result = session.run_step()
        assert result["ready"]
        assert result["window_end_timestamp"] == ts[-1]
        assert session.eeg_buffer.shape[1] == 512
        offset += n
    assert policy.block_id == 2
    assert actions[-1]["master_volume"] == .03
    # Retained samples must never survive a real gap.
    now[0] += 1
    chunk[4] += 1
    assert not session.run_step()["ready"]
    assert session.eeg_buffer.shape[1] == 0
    assert actions[-1]["master_volume"] == 0
