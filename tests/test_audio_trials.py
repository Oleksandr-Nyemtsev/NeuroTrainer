import json
import pytest
from src.core.audio_experiment import AudioExperiment
from src.data.session_journal import SessionJournal
from src.data.audio_trials import build_trials


def journal(tmp_path, enabled=True):
    log = SessionJournal(tmp_path, audio_enabled=enabled, metadata={"policy": "AudioExperiment"})
    policy = AudioExperiment()
    for t in range(0, 182, 2):
        action = policy.choose(1, {"ready": True}, t)
        # Deliberate known step in synthetic alpha/beta after sound begins.
        alpha = 20 if t <= 60 else 25
        obs = {"ready": True, "window_start_timestamp": t-1.99, "window_end_timestamp": t,
               "state": {"Delta":20, "Theta":20, "Alpha":alpha, "Beta":40-alpha, "Gamma":20},
               "action": action, "audio_experiment": policy.describe()}
        log.record(obs, {"started_at":t+.01, "completed_at":t+.02})
    log.close("completed")
    return log.path


def mutate(path, fn):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    fn(rows)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_known_delta_no_future_baseline(tmp_path):
    result = build_trials(journal(tmp_path))
    sound = [r for r in result["examples"] if r["comparison"] == "sound_after_silence"][0]
    assert sound["delta_percentage_points"]["Alpha"] == 5
    assert max(sound["before_observation_ids"]) < min(sound["after_observation_ids"])
    assert not sound["causal_effect"]
    assert result["rejected_blocks"]["unfinished_or_interrupted_block"] == 1


def test_dry_run_rejected(tmp_path):
    with pytest.raises(ValueError, match="dry runs"):
        build_trials(journal(tmp_path, False))


def test_muted_sound_is_not_training_data(tmp_path):
    path = journal(tmp_path)
    def edit(rows):
        for r in rows:
            if r['type'] == 'observation':
                r['effective_targets']['master_volume'] = 0
    mutate(path, edit)
    result = build_trials(path)
    assert not any(r['comparison'] == 'sound_after_silence' for r in result['examples'])
    assert 'audio_parameters_mismatch' in result['rejected_blocks']


def test_quality_interruption_rejects_block(tmp_path):
    path = journal(tmp_path)
    mutate(path, lambda rows: rows[40]['observation'].update(ready=False))
    assert not any(r['comparison'] == 'sound_after_silence' for r in build_trials(path)['examples'])


def test_overlapping_windows_do_not_multiply_evidence(tmp_path):
    path = journal(tmp_path)
    def edit(rows):
        for r in rows:
            if r['type'] == 'observation' and 100 <= r['observation']['window_end_timestamp'] < 120:
                r['observation']['window_start_timestamp'] = 100
                r['observation']['window_end_timestamp'] = 102
    mutate(path, edit)
    result = build_trials(path)
    assert not any(r['comparison'] == 'sound_after_silence' for r in result['examples'])
