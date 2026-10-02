import numpy as np
import pytest
from src.signal.alpha_tracker import AlphaTracker
from src.signal.slow_wave_detector import SlowWaveDetector
from src.core.brain_state import BrainState
from src.core.stimulation_policy import StimulationPolicy


@pytest.mark.parametrize("value", [0.0, 12.0, np.nan, np.inf])
def test_undefined_rhythm(value):
    eeg = np.full((4, 2560), value)
    alpha = AlphaTracker().analyze(eeg)
    slow = SlowWaveDetector().analyze(eeg)
    assert alpha["iaf"] is None
    assert not alpha["rhythm_defined"] and not slow["rhythm_defined"]
    assert alpha["alpha_stability"] == slow["slow_wave_stability"] == 0.0
    for stage in ["Wake", "N3"]:
        state = BrainState(sleep_probs={stage: 1.}, **{k:v for k,v in alpha.items() if k != "rhythm_defined"},
                           **{k:v for k,v in slow.items() if k != "rhythm_defined"})
        assert not StimulationPolicy().decide(state).enabled


def test_existing_synthetic_rhythms_remain_defined():
    t = np.arange(256*20)/256
    tracker = AlphaTracker()
    alpha = tracker.analyze(np.tile(np.sin(2*np.pi*10*t), (4, 1)))
    slow = SlowWaveDetector().analyze(np.tile(np.sin(2*np.pi*.8*t), (4, 1)))
    assert alpha["iaf"] == pytest.approx(10.)
    assert alpha["rhythm_defined"] and slow["rhythm_defined"]
    assert tracker.analyze(np.zeros((4, len(t))))["iaf"] is None


def test_numerical_amplitude_floor():
    t = np.arange(2560)/256
    eeg = np.tile(1e-9*np.sin(2*np.pi*10*t), (4,1))
    assert not AlphaTracker().analyze(eeg)["rhythm_defined"]
    assert not SlowWaveDetector().analyze(eeg)["rhythm_defined"]


def test_policy_rejects_zero_amplitude_even_with_claimed_stability():
    for stage in ["Wake", "N3"]:
        state = BrainState(sleep_probs={stage:1.}, iaf=10., alpha_stability=1., slow_wave_stability=1.)
        assert not StimulationPolicy().decide(state).enabled
