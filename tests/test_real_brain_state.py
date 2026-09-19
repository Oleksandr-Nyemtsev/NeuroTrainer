import numpy as np

from src.config import SLEEP_EDF_DIR, SLEEP_MODEL_PATH
from src.models.sleep_model import SleepModel
from src.core.brain_state_estimator import BrainStateEstimator


files = sorted(
    SLEEP_EDF_DIR.glob("*.npz")
)

if len(files) == 0:
    raise RuntimeError(
        "No processed Sleep-EDF files found."
    )


file = files[0]

print(
    "Using recording:",
    file.name
)


with np.load(file) as data:

    X = data["X"]
    y = data["y"]


index = 100

eeg_for_model = X[index]


# Тут для тесту беремо той самий 2-канальний
# сигнал як raw EEG
eeg_raw = eeg_for_model


sleep_model = SleepModel(
    model_path=SLEEP_MODEL_PATH
)


estimator = BrainStateEstimator(
    sleep_model=sleep_model,
    fs=100,
)


state = estimator.estimate(
    eeg_for_model=eeg_for_model,
    eeg_raw=eeg_raw,
    signal_quality=0.95,
    minutes_elapsed=18.5,
)


LABELS = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]


true_stage = LABELS[
    int(y[index])
]


print()
print(
    "True stage:",
    true_stage
)

print()

print(
    "Dominant stage:",
    state.dominant_stage()
)

print()

print(
    "Sleep probabilities:"
)

for stage, probability in state.sleep_probs.items():

    print(
        f"{stage}: "
        f"{probability:.3f}"
    )


print()
print(
    "IAF:",
    state.iaf
)

print(
    "Alpha amplitude:",
    state.alpha_amplitude
)

print(
    "Alpha phase:",
    state.alpha_phase
)

print(
    "Alpha stability:",
    state.alpha_stability
)


print()
print(
    "Slow-wave amplitude:",
    state.slow_wave_amplitude
)

print(
    "Slow-wave phase:",
    state.slow_wave_phase
)

print(
    "Slow-wave stability:",
    state.slow_wave_stability
)


print()
print(
    "Signal quality:",
    state.signal_quality
)

print(
    "Minutes elapsed:",
    state.minutes_elapsed
)

print()
print(
    "REAL BRAIN STATE TEST OK"
)
