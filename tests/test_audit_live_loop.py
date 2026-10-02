import logging
from types import SimpleNamespace
import numpy as np
import pytest
import src.core.session as module


@pytest.fixture
def setup(monkeypatch):
    now = [1000.]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module, "BoardShim", SimpleNamespace(get_eeg_channels=lambda _: [0,1,2,3], get_timestamp_channel=lambda _: 4))
    actions = []
    muse = SimpleNamespace(get_data=lambda: None)
    session = module.NeuroSession(muse, SimpleNamespace(apply_action=actions.append, target_master_volume=.1))
    return session, muse, actions, now


def chunk(n, end=1000.):
    eeg = np.tile(20*np.sin(2*np.pi*10*np.arange(n)/256), (4,1))
    return np.vstack([eeg, end-(n-1-np.arange(n))/256])


def test_backlog_drops_oldest_and_logs_timestamps(setup, caplog):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(2560)
    with caplog.at_level(logging.INFO):
        result = session.run_step()
    assert result["ready"]
    assert result["dropped_samples"] == 2048
    assert result["dropped_windows"] == 4
    assert result["window_start_timestamp"] == pytest.approx(1000-511/256)
    assert result["window_end_timestamp"] == 1000
    assert result["latency_seconds"] == 0
    assert session.eeg_buffer.shape[1] <= 512
    assert "dropped_windows" in caplog.text


@pytest.mark.parametrize("problem", ["gap", "duplicate", "nan", "stale", "future"])
def test_bad_timestamps_mute_audio(setup, problem):
    session, muse, actions, now = setup
    data = chunk(512)
    if problem == "gap": data[4, 200:] += .1
    elif problem == "duplicate": data[4,200] = data[4,199]
    elif problem == "nan": data[4,200] = np.nan
    elif problem == "stale": data[4] -= 10
    else: data[4] += 10
    muse.get_data = lambda: data
    assert not session.run_step()["ready"]
    assert actions[-1] == {"master_volume": 0.}
    assert session.dropped_samples == 512


def test_gap_between_chunks_is_not_hidden(setup):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(256, end=999.)
    assert not session.run_step()["ready"]
    muse.get_data = lambda: chunk(256, end=1000.1)
    result = session.run_step()
    assert not result["ready"] and "gap" in result["reason"]


def test_partial_chunks_form_one_window(setup):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(256, end=999.)
    assert not session.run_step()["ready"]
    muse.get_data = lambda: chunk(256, end=1000.)
    assert session.run_step()["ready"]
    assert session.dropped_samples == 0


def test_processing_delay_is_included_in_latency(setup, monkeypatch):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(512)
    process = module.process_eeg
    def delayed(*a, **k):
        now[0] += 5
        return process(*a, **k)
    monkeypatch.setattr(module, "process_eeg", delayed)
    assert not session.run_step()["ready"]
    assert actions[-1] == {"master_volume":0.}


def test_loop_subtracts_processing_time(monkeypatch, setup):
    session, muse, actions, now = setup
    clock = [0.]
    sleeps = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module.time, "sleep", sleep)
    def step():
        clock[0] += .5
        return {"ready": True}
    session.run_step = step
    muse.connect = muse.start = lambda: True
    muse.stop = session.audio.start = session.audio.stop = lambda: None
    session.run(duration_seconds=4, step_seconds=2)
    assert sleeps == [1.5, 1.5]
