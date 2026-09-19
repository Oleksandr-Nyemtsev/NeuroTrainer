import numpy as np

from src.signal.slow_wave_detector import SlowWaveDetector


FS = 256
DURATION = 30
TRUE_SLOW_WAVE = 0.8


t = np.arange(
    0,
    DURATION,
    1 / FS,
)


channel_1 = np.sin(
    2 * np.pi * TRUE_SLOW_WAVE * t
)

channel_2 = 0.9 * np.sin(
    2 * np.pi * TRUE_SLOW_WAVE * t + 0.1
)


rng = np.random.default_rng(42)

channel_1 += 0.10 * rng.normal(
    size=len(t)
)

channel_2 += 0.10 * rng.normal(
    size=len(t)
)


eeg = np.vstack(
    [
        channel_1,
        channel_2,
    ]
)


detector = SlowWaveDetector(
    fs=FS,
    low=0.5,
    high=1.2,
)


result = detector.analyze(
    eeg
)


print(
    "True slow-wave frequency:",
    TRUE_SLOW_WAVE,
)

print(
    "Slow-wave amplitude:",
    result["slow_wave_amplitude"],
)

print(
    "Slow-wave phase:",
    result["slow_wave_phase"],
)

print(
    "Slow-wave stability:",
    result["slow_wave_stability"],
)
