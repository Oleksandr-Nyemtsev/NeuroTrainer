from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

SLEEP_EDF_DIR = (
    DATA_DIR
    / "processed"
    / "sleep_edf"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

SLEEP_MODEL_PATH = (
    MODEL_DIR
    / "sleep_cnn_best.pt"
)