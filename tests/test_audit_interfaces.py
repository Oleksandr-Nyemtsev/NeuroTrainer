from dataclasses import asdict
from types import SimpleNamespace
import runpy
import numpy as np
import pytest
from src.core.brain_state import BrainState
from src.core.brain_state_estimator import BrainStateEstimator
from src.core.adaptive_audio_controller import AdaptiveAudioController
from src.core.stimulation_policy import StimulationPolicy
from src.data.sleep_dataset import split_subjects


def signal():
    t = np.arange(2560)/256
    return np.tile(20*np.sin(2*np.pi*10*t) + 5*np.sin(2*np.pi*.8*t), (2,1))


def test_estimator_controller_contract():
    probs = {"Wake":.8, "N1":.1, "N2":.05, "N3":.03, "REM":.02}
    estimator = BrainStateEstimator(SimpleNamespace(predict=lambda eeg: probs))
    state = estimator.estimate(signal(), signal(), signal_quality=.95, minutes_elapsed=2.)
    assert isinstance(state, BrainState) and state.is_valid()
    assert state.sleep_probs == probs and state.dominant_stage() == "Wake"
    action = AdaptiveAudioController().choose_action(state)
    assert action.master_volume == .65
    assert all(np.isfinite(v) for v in asdict(action).values())


@pytest.mark.parametrize("probs", [{}, {"Wake":np.nan}, {"N2":np.inf}])
def test_invalid_model_output_mutes_controller(probs):
    state = BrainStateEstimator(SimpleNamespace(predict=lambda eeg: probs)).estimate(signal(), signal())
    assert state.is_valid() and state.signal_quality == 0
    assert AdaptiveAudioController().choose_action(state).master_volume == 0
    assert not StimulationPolicy().decide(state).enabled


def test_flatline_never_reaches_model():
    def unexpected(eeg):
        pytest.fail("Flatline reached inference")
    state = BrainStateEstimator(SimpleNamespace(predict=unexpected)).estimate(signal(), np.zeros((2,512)))
    assert state.signal_quality == 0 and state.iaf is None
    assert AdaptiveAudioController().choose_action(state).master_volume == 0


def test_mutated_nonfinite_state_is_safe():
    state = BrainState()
    state.alpha_phase = np.nan
    assert AdaptiveAudioController().choose_action(state).master_volume == 0
    assert not StimulationPolicy().decide(state).enabled


def test_multiple_nights_stay_in_same_subject_split():
    recordings = [{"subject_id":s, "recording_id":night} for s in range(20) for night in (1,2)]
    groups = split_subjects(recordings)
    assert set.union(*groups) == set(range(20))
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert groups == split_subjects(recordings)
    for s in range(20):
        assert sum(s in group for group in groups) == 1


def test_repaired_demos_with_fake_audio(monkeypatch):
    import src.audio.engine as engine
    import time
    class FakeAudio:
        def __init__(self, carrier_frequency=180., beat_frequency=8., binaural_volume=.02, device=None):
            self.carrier_frequency = carrier_frequency
            self.beat_frequency = beat_frequency
        def start(self): pass
        def stop(self): pass
        def apply_action(self, action): pass
    monkeypatch.setattr(engine, "AudioEngine", FakeAudio)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    runpy.run_path("tests/test_closed_loop.py", run_name="__main__")
    runpy.run_path("tests/test_adaptive_session.py", run_name="__main__")
    result = runpy.run_path("tests/test_brain_state_estimator.py", run_name="__main__")
    assert result["state"].is_valid()
    assert result["eeg_for_model"].shape == (2,3000)
