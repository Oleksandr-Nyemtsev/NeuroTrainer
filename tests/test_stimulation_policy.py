import numpy as np

from src.config import SLEEP_EDF_DIR, SLEEP_MODEL_PATH
from src.models.sleep_model import SleepModel
from src.core.brain_state_estimator import BrainStateEstimator
from src.core.stimulation_policy import StimulationPolicy


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
    minutes_elapsed=25.0,
)


policy = StimulationPolicy()


decision = policy.decide(
    state
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
print("True stage:")
print(
    true_stage
)

print()
print("Dominant stage:")
print(
    state.dominant_stage()
)

print()
print("Alpha reliable:")
print(
    state.alpha_reliable()
)

print("Slow-wave reliable:")
print(
    state.slow_wave_reliable()
)

print()
print("Stimulation mode:")
print(
    decision.mode.value
)

print("Enabled:")
print(
    decision.enabled
)

print("Target frequency:")
print(
    decision.target_frequency
)

print("Target phase:")
print(
    decision.target_phase
)

print("Pulse gain:")
print(
    decision.pulse_gain
)

print("Reason:")
print(
    decision.reason
)

print()
print(
    "STIMULATION POLICY TEST OK"
)
