import numpy as np

from src.config import SLEEP_EDF_DIR, SLEEP_MODEL_PATH
from src.models.sleep_model import SleepModel
from src.core.brain_state_estimator import BrainStateEstimator
from src.core.adaptive_audio_controller import AdaptiveAudioController


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


controller = AdaptiveAudioController(
    session_minutes=45,
)


state = estimator.estimate(
    eeg_for_model=eeg_for_model,
    eeg_raw=eeg_raw,
    signal_quality=0.95,
    minutes_elapsed=25.0,
)


action = controller.choose_action(
    state
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
print("Alpha reliable:")
print(
    state.alpha_reliable()
)

print("Slow wave reliable:")
print(
    state.slow_wave_reliable()
)

print()
print("Session phase:")
print(
    controller.phase.value
)

print()
print("Audio action:")
print(
    action
)

print()
print(
    "BRAIN TO AUDIO TEST OK"
)
