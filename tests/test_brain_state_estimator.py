import numpy as np
from scipy.signal import resample_poly

from src.core.brain_state_estimator import BrainStateEstimator


class DummySleepModel:
    def predict(self, eeg):
        return {"Wake": 0.1, "N1": 0.2, "N2": 0.5, "N3": 0.15, "REM": 0.05}


FS = 256
DURATION = 30

t = np.arange(
    0,
    DURATION,
    1 / FS,
)


# Synthetic EEG:
# alpha 10 Hz
# + slow wave 0.8 Hz
# + трохи noise

rng = np.random.default_rng(42)

channel_1 = (
    0.8 * np.sin(
        2 * np.pi * 10.0 * t
    )
    +
    0.5 * np.sin(
        2 * np.pi * 0.8 * t
    )
    +
    0.10 * rng.normal(
        size=len(t)
    )
)

channel_2 = (
    0.7 * np.sin(
        2 * np.pi * 10.0 * t + 0.1
    )
    +
    0.45 * np.sin(
        2 * np.pi * 0.8 * t + 0.05
    )
    +
    0.10 * rng.normal(
        size=len(t)
    )
)


eeg_raw = np.vstack(
    [
        channel_1,
        channel_2,
    ]
)


# Для тесту SleepCNN беремо
# 3000 samples, як у нашій моделі
eeg_for_model = resample_poly(eeg_raw, 25, 64, axis=1)


model = DummySleepModel()


estimator = BrainStateEstimator(
    sleep_model=model,
    fs=FS,
)


state = estimator.estimate(
    eeg_for_model=eeg_for_model,
    eeg_raw=eeg_raw,
    signal_quality=0.95,
    minutes_elapsed=18.5,
)


print()
print("Dominant stage:")
print(
    state.dominant_stage()
)

print()
print("Sleep probabilities:")
print(
    state.sleep_probs
)

print()
print("IAF:")
print(
    state.iaf
)

print()
print("Alpha amplitude:")
print(
    state.alpha_amplitude
)

print()
print("Alpha stability:")
print(
    state.alpha_stability
)

print()
print("Slow-wave amplitude:")
print(
    state.slow_wave_amplitude
)

print()
print("Slow-wave stability:")
print(
    state.slow_wave_stability
)

print()
print("Signal quality:")
print(
    state.signal_quality
)

print()
print("Minutes elapsed:")
print(
    state.minutes_elapsed
)

print()
print(
    "BRAIN STATE ESTIMATOR TEST OK"
)
