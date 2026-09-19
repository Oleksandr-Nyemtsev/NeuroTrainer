import numpy as np
import torch

from src.core.brain_state_estimator import BrainStateEstimator


class DummySleepModel(torch.nn.Module):

    def forward(self, x):

        batch_size = x.shape[0]

        # Просто тестові logits для 5 класів:
        # Wake, N1, N2, N3, REM
        logits = torch.tensor(
            [[0.5, 1.0, 2.0, 0.8, 0.2]],
            dtype=torch.float32,
            device=x.device,
        )

        return logits.repeat(
            batch_size,
            1,
        )


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
eeg_for_model = eeg_raw[
    :,
    :3000
]


model = DummySleepModel()


estimator = BrainStateEstimator(
    sleep_model=model,
    fs=FS,
    device="cpu",
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
