import numpy as np

from src.config import SLEEP_EDF_DIR, SLEEP_MODEL_PATH
from src.models.sleep_model import SleepModel


# Знаходимо перший готовий .npz
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


# Завантажуємо запис
with np.load(file) as data:

    X = data["X"]
    y = data["y"]


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# Завантажуємо справжню CNN
model = SleepModel(
    model_path=SLEEP_MODEL_PATH
)


# Беремо одну 30-секундну епоху
index = 100

eeg = X[index]

true_label = int(
    y[index]
)


# Прогноз
probs = model.predict(
    eeg
)


LABELS = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]


predicted_stage = max(
    probs,
    key=probs.get
)


print()
print(
    "Epoch index:",
    index
)

print(
    "True stage:",
    LABELS[true_label]
)

print()

print(
    "Model probabilities:"
)

for stage, probability in probs.items():

    print(
        f"{stage}: "
        f"{probability:.3f}"
    )


print()
print(
    "Predicted stage:",
    predicted_stage
)
