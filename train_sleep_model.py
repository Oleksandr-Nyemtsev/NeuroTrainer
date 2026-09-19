from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.sleep_dataset import (
    SleepDataset,
    find_recordings,
    split_subjects,
    load_subject_data,
)

from src.models.sleep_cnn import SleepCNN

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_edf"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )

    accuracy = (
        correct / total * 100
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    average_loss = (
        total_loss / len(loader)
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
    )


def main():

    print("=" * 60)
    print("NeuroTrainer - Sleep CNN training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    print()

    recordings = find_recordings(
        DATA_DIR
    )

    print(
        f"Recordings found: {len(recordings)}"
    )

    if len(recordings) < 3:
        raise RuntimeError(
            "Not enough processed recordings."
        )

    (
        train_subjects,
        val_subjects,
        test_subjects,
    ) = split_subjects(
        recordings,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )

    print(
        f"Train subjects: {len(train_subjects)}"
    )

    print(
        f"Validation subjects: {len(val_subjects)}"
    )

    print(
        f"Test subjects: {len(test_subjects)}"
    )

    print()

    print("Loading TRAIN...")
    X_train, y_train = load_subject_data(
        recordings,
        train_subjects,
    )

    print("Loading VALIDATION...")
    X_val, y_val = load_subject_data(
        recordings,
        val_subjects,
    )

    print("Loading TEST...")
    X_test, y_test = load_subject_data(
        recordings,
        test_subjects,
    )

    print()
    print(
        "Train:",
        X_train.shape,
        y_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape,
    )


    train_dataset = SleepDataset(
        X_train,
        y_train,
    )

    val_dataset = SleepDataset(
        X_val,
        y_val,
    )

    test_dataset = SleepDataset(
        X_test,
        y_test,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


    model = SleepCNN().to(
        DEVICE
    )


    class_counts = np.bincount(
        y_train,
        minlength=5,
    )

    class_weights = (
        len(y_train)
        / (
            5
            * class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print(
        "Class counts:",
        class_counts,
    )

    print(
        "Class weights:",
        class_weights,
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    best_val_f1 = -1.0

    best_model_path = (
        MODEL_DIR
        / "sleep_cnn_best.pt"
    )


    print()
    print("Training started...")
    print()


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0


        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch,
            )

            loss.backward()

            optimizer.step()


            train_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == y_batch
            ).sum().item()

            train_total += (
                y_batch.size(0)
            )


        train_loss /= len(
            train_loader
        )

        train_accuracy = (
            train_correct
            / train_total
            * 100
        )


        (
            val_loss,
            val_accuracy,
            val_f1,
        ) = evaluate(
            model,
            val_loader,
            criterion,
        )


        if DEVICE.type == "cuda":
            torch.cuda.synchronize()


        epoch_time = (
            time.time()
            - epoch_start
        )


        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Time {epoch_time:.1f}s | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_accuracy:.1f}% | "
            f"Val loss {val_loss:.4f} | "
            f"Val acc {val_accuracy:.1f}% | "
            f"Val F1 {val_f1:.3f}"
        )


        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                "  Best model saved:",
                best_model_path.name,
            )


    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Best validation F1: "
        f"{best_val_f1:.3f}"
    )


    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )


    (
        test_loss,
        test_accuracy,
        test_f1,
    ) = evaluate(
        model,
        test_loader,
        criterion,
    )


    print()
    print("FINAL TEST")

    print(
        f"Loss: {test_loss:.4f}"
    )

    print(
        f"Accuracy: {test_accuracy:.1f}%"
    )

    print(
        f"Macro F1: {test_f1:.3f}"
    )


    print()
    print(
        "Model saved to:",
        best_model_path,
    )


if __name__ == "__main__":
    main()