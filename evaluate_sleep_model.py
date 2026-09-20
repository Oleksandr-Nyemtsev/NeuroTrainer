from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import SLEEP_EDF_DIR, SLEEP_MODEL_PATH
from src.data.sleep_dataset import (
    find_recordings,
    split_subjects,
    load_subject_data,
)
from src.models.sleep_cnn import SleepCNN


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    files = find_recordings(SLEEP_EDF_DIR)

    train_subjects, val_subjects, test_subjects = split_subjects(files)

    print("Train subjects:", len(train_subjects))
    print("Validation subjects:", len(val_subjects))
    print("Test subjects:", len(test_subjects))

    print("\nLoading test data...")

    X_test, y_test = load_subject_data(files, test_subjects)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    model = SleepCNN().to(device)

    state_dict = torch.load(
        SLEEP_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    batch_size = 256

    predictions = []

    print("\nRunning inference...")

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = start + batch_size

            batch = X_tensor[start:end].to(device)

            logits = model(batch)

            pred = torch.argmax(logits, dim=1)

            predictions.extend(pred.cpu().numpy())

    predictions = np.array(predictions)

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=STAGE_NAMES,
            digits=3,
        )
    )

    print("========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    cm = confusion_matrix(y_test, predictions)

    print("\nRows = TRUE")
    print("Columns = PREDICTED\n")

    print("        Wake     N1      N2      N3     REM")

    for name, row in zip(STAGE_NAMES, cm):
        values = " ".join(f"{value:7d}" for value in row)
        print(f"{name:5s} {values}")


if __name__ == "__main__":
    main()