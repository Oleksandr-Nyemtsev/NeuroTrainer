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
    assert sleeps == [1] * 5 + [1.5, 1.5]
    assert clock[0] == 9.0


@pytest.mark.parametrize("problem", ["gap", "duplicate", "backward"])
def test_timestamp_diagnostic_counts(setup, problem):
    session, muse, actions, now = setup
    data = chunk(512)
    if problem == "gap":
        data[4, 200:] += .1
    elif problem == "duplicate":
        data[4, 200] = data[4, 199]
    else:
        data[4, 200] = data[4, 199] - .001
    muse.get_data = lambda: data
    result = session.run_step()
    d = result["timestamp_diagnostics"]
    assert not result["ready"]
    assert d["received_samples"] == 512
    assert d["boundary_interval_ms"] is None
    assert d["expected_interval_ms"] == pytest.approx(1000 / 256)
    assert d["gap_threshold_ms"] == pytest.approx(1500 / 256)
    assert d["duplicate_count"] == int(problem == "duplicate")
    assert d["backward_count"] == int(problem == "backward")
    assert d["gap_count"] == 1
    if problem == "gap":
        assert d["max_interval_ms"] == pytest.approx(100 + 1000 / 256)
    assert actions[-1] == {"master_volume": 0.}


def test_timestamp_boundary_and_diagnostic_reset(setup):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(256, end=999.)
    session.run_step()
    muse.get_data = lambda: chunk(256, end=1000.1)
    result = session.run_step()
    d = result["timestamp_diagnostics"]
    assert d["boundary_interval_ms"] == pytest.approx(100 + 1000 / 256)
    assert d["gap_count"] == 1
    muse.get_data = lambda: None
    assert "timestamp_diagnostics" not in session.run_step()


def test_valid_stream_keeps_timestamp_diagnostics(setup):
    session, muse, actions, now = setup
    muse.get_data = lambda: chunk(512)
    result = session.run_step()
    assert result["ready"]
    d = result["timestamp_diagnostics"]
    assert d["gap_count"] == d["duplicate_count"] == d["backward_count"] == 0
    assert d["min_interval_ms"] == d["max_interval_ms"] == 1000 / 256


def test_packet_ids_recorded_even_when_timestamps_rejected(setup, tmp_path):
    import json
    from src.data.session_journal import SessionJournal
    session, muse, actions, now = setup
    journal = SessionJournal(tmp_path, audio_enabled=False, metadata={})
    session.journal = journal
    muse.package_num_channel = 5
    data = chunk(504)
    data[4, 200:] += .1
    data = np.vstack([data, np.repeat(np.arange(42), 12)])
    muse.get_data = lambda: data
    result = session.run_step()
    assert not result['ready']
    assert result['timestamp_diagnostics']['gap_count'] == 1
    assert result['packet_diagnostics']['forward_jump_count'] == 0
    assert result['packet_diagnostics']['consecutive_packet_steps'] == 41
    assert actions[-1] == {'master_volume': 0.}
    journal.close('completed')
    records = [json.loads(s) for s in journal.path.read_text().splitlines()]
    assert records[1]['observation']['packet_diagnostics'] == result['packet_diagnostics']


def test_bilingual_message_precedes_stream_start(setup, capsys, monkeypatch):
    waits = []
    monkeypatch.setattr(module.time, "sleep", waits.append)
    session, muse, actions, now = setup
    muse.connect = lambda: True
    def start():
        printed = capsys.readouterr().out
        assert waits == [1] * 5
        assert [line.rsplit(': ', 1)[-1] for line in printed.splitlines() if line.startswith('Starting in')] == ['5', '4', '3', '2', '1']
        assert 'Muse is about to start. Please get ready.' in printed
        assert 'Muse зараз почне запис. Будь ласка, підготуйтеся.' in printed
        return False
    muse.start = start
    muse.stop = lambda: None
    assert session.run(duration_seconds=1) == 'stream_start_failed'


def test_countdown_interrupt_disconnects_without_starting_stream(setup, monkeypatch):
    session, muse, actions, now = setup
    events = []
    muse.connect = lambda: True
    muse.start = lambda: events.append('start')
    muse.stop = lambda: events.append('stop')
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(module.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        session.run(duration_seconds=1)
    assert events == ['stop']
