import numpy as np

from src.signal.multichannel import process_eeg
from src.signal.state import build_state
from src.signal.metrics import relaxation_score
from src.core.controller import choose_audio_action
from src.audio.engine import AudioEngine


fs = 256
seconds = 2

t = np.arange(0, seconds, 1 / fs)

# Штучний EEG: 4 канали
tp9 = 20 * np.sin(2 * np.pi * 10 * t)
af7 = 18 * np.sin(2 * np.pi * 8 * t)
af8 = 15 * np.sin(2 * np.pi * 12 * t)
tp10 = 22 * np.sin(2 * np.pi * 6 * t)

eeg = np.array([
    tp9,
    af7,
    af8,
    tp10
])

results = process_eeg(eeg, fs=fs)

state = build_state(results)

score = relaxation_score(state)

action = choose_audio_action(score)

print("STATE:")
print(state)

print("\nSCORE:")
print(score)

print("\nACTION:")
print(action)

audio = AudioEngine(
    carrier_frequency=440,
    beat_frequency=8,
    binaural_volume=0.10,
    device=3
)

audio.apply_action(action)

print("\nAUDIO PARAMETERS:")
print("Carrier:", audio.carrier_frequency)
print("Beat:", audio.beat_frequency)


def test_closed_loop_without_audio_output():
    assert np.isfinite(score)
    assert action["beat_frequency"] in (6, 8, 10)
    assert audio.stream is None
    assert audio.target_beat_frequency == action["beat_frequency"]
