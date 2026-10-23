import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.core.neural_audio import ACTIONS, FEATURES, ShadowController, encode, candidate_parameters
from train_audio_response import read_trials, train


def observation(t=1000):
    return {"ready": True, "window_end_timestamp": t,
            "state": dict(zip(FEATURES[:5], [20.] * 5)),
            "sleep": {"ready": True, "probabilities":
                      dict(zip(FEATURES[5:10], [.6, .1, .1, .1, .1]))}}


def trial(session, action, rating):
    return {"session_id": session, "trial_id": action, "action_id": action,
            "before": observation(), "previous_action": {"master_volume": 0.},
            "rating_source": "self_report", "rating": rating, "delivered": True,
            "delivered_parameters": candidate_parameters(action),
            "action_started_at": 1001., "action_ended_at": 1031., "rated_at": 1032.}


@pytest.fixture
def trained(tmp_path):
    # Artificial ratings test software learning only, never clinical efficacy.
    rows = [trial(session, a, rating) for session in ("train", "validation")
            for a, rating in zip(ACTIONS, [-.8, -.2, .4, .8])]
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    checkpoint = train(read_trials(path), ["validation"], epochs=200)
    output = tmp_path / "response.pt"
    torch.save(checkpoint, output)
    return output, checkpoint, rows


def test_learning_roundtrip_and_decision_interval(trained):
    path, checkpoint, _ = trained
    assert checkpoint["validation_mae"] < checkpoint["constant_baseline_mae"] / 2
    assert not set(checkpoint["training_sessions"]) & set(checkpoint["validation_sessions"])
    controller = ShadowController(path)
    answer = controller.suggest(observation(), {"master_volume": 0.})
    assert answer["action_id"] == "beat_10" and answer["applied"] is False
    assert not controller.suggest(observation(1002), {})["ready"]
    assert controller.suggest(observation(1030), {})["ready"]
    assert not controller.suggest(observation(1020), {})["ready"]
    assert not controller.suggest({"ready": False}, {})["ready"]


@pytest.mark.parametrize("field,value", [("rating", float("nan")), ("delivered", False),
    ("rating_source", "REM_probability"), ("action_started_at", 999.),
    ("action_ended_at", 1002.), ("delivered_parameters", {})])
def test_reject_invalid_or_automatic_rewards(tmp_path, field, value):
    row = trial("s1", "silence", 0.)
    row[field] = value
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError): read_trials(path)


def test_bad_features_and_checkpoint(trained, tmp_path):
    bad = observation()
    bad["sleep"]["probabilities"]["REM"] = float("nan")
    with pytest.raises(ValueError): encode(bad, {})
    _, checkpoint, _ = trained
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["actions"].reverse()
    path = tmp_path / "bad.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError): ShadowController(path)


def test_session_split_and_action_coverage(trained, tmp_path):
    _, _, rows = trained
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    parsed = read_trials(path)
    with pytest.raises(ValueError): train(parsed, ["unknown"])
    with pytest.raises(ValueError): train(parsed, ["train", "validation"])
    with pytest.raises(ValueError): train([r for r in parsed if r[2] != 0], ["validation"])
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]))
    with pytest.raises(ValueError): read_trials(path)


def test_shadow_never_changes_audio_and_is_journaled(trained, monkeypatch, tmp_path):
    import src.core.session as module
    from src.data.session_journal import SessionJournal
    path, _, _ = trained
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    monkeypatch.setattr(module, "BoardShim", SimpleNamespace(
        get_eeg_channels=lambda _: [0, 1, 2, 3], get_timestamp_channel=lambda _: 4))
    eeg = np.tile(20 * np.sin(2 * np.pi * 10 * np.arange(512) / 256), (4, 1))
    data = np.vstack([eeg, 1000 - (511 - np.arange(512)) / 256])
    muse = SimpleNamespace(get_data=lambda: data)
    observer = SimpleNamespace(update=lambda *args: observation()["sleep"], reset=lambda: None)
    applied = []
    audio = SimpleNamespace(apply_action=applied.append, target_master_volume=0.)
    journal = SessionJournal(tmp_path / "logs", audio_enabled=False, metadata={})
    session = module.NeuroSession(muse, audio, sleep_observer=observer,
                                  shadow_controller=ShadowController(path), journal=journal)
    result = session.run_step()
    assert result["controller_shadow"]["applied"] is False
    assert applied == [result["action"]]
    assert "proposed_parameters" not in applied[0]
    muse.get_data = lambda: None
    session.run_step()
    assert session.shadow_controller.last_time is None
    journal.close("completed")
    logged = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert logged[1]["observation"]["controller_shadow"]["mode"] == "shadow_only"
    assert logged[1]["effective_targets"] == {"master_volume": 0.}
