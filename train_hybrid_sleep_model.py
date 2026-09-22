import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.config import SLEEP_EDF_DIR, MODEL_DIR
from src.data.sleep_dataset import (
    find_recordings,
    split_subjects,
    load_subject_data,
)
from src.models.hybrid_crg_sleep import HybridCRGSleep


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 5

STAGE_NAMES = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]

MODEL_PATH = MODEL_DIR / "hybrid_crg_sleep_best.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y, num_classes):
    counts = np.bincount(
        y,
        minlength=num_classes,
    )

    total = counts.sum()

    weights = total / (
        num_classes * counts
    )

    return counts, weights


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            total_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    all_targets = np.asarray(
        all_targets
    )

    all_predictions = np.asarray(
        all_predictions
    )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = (
        all_targets
        == all_predictions
    ).mean()

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_targets,
        all_predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("HYBRID CNN + SPD SLEEP TRAINING")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("\nFinding recordings...")

    files = find_recordings(
        SLEEP_EDF_DIR
    )

    print(
        "Recordings found:",
        len(files),
    )

    train_subjects, val_subjects, test_subjects = (
        split_subjects(files)
    )

    print(
        "Train subjects:",
        len(train_subjects),
    )

    print(
        "Validation subjects:",
        len(val_subjects),
    )

    print(
        "Test subjects:",
        len(test_subjects),
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading TRAIN data...")

    X_train, y_train = load_subject_data(
        files,
        train_subjects,
    )

    print("\nLoading VALIDATION data...")

    X_val, y_val = load_subject_data(
        files,
        val_subjects,
    )

    print("\nLoading TEST data...")

    X_test, y_test = load_subject_data(
        files,
        test_subjects,
    )

    print("\nShapes:")

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

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_counts, class_weights = (
        calculate_class_weights(
            y_train,
            NUM_CLASSES,
        )
    )

    print(
        "\nClass counts:",
        class_counts,
    )

    print(
        "Class weights:",
        class_weights,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train
        ).float(),
        torch.from_numpy(
            y_train
        ).long(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(
            X_val
        ).float(),
        torch.from_numpy(
            y_val
        ).long(),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(
            X_test
        ).float(),
        torch.from_numpy(
            y_test
        ).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = HybridCRGSleep(
        num_classes=NUM_CLASSES
    ).to(device)

    print("\nModel:")
    print(model)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{trainable_parameters:,}",
    )

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_f1 = -1.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nTraining started...\n")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start_time = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            # Catch numerical instability early.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected. "
                    "Possible SPD/eigendecomposition instability."
                )

            loss.backward()

            # Useful because this model differentiates
            # through covariance + eigendecomposition.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions
                == y_batch
            ).sum().item()

            total += y_batch.size(0)

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            correct
            / total
        )

        (
            val_loss,
            val_accuracy,
            val_f1,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_f1
        )

        elapsed = (
            time.time()
            - start_time
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Time {elapsed:.1f}s"
            f" | Train loss {train_loss:.4f}"
            f" | Train acc {train_accuracy * 100:.1f}%"
            f" | Val loss {val_loss:.4f}"
            f" | Val acc {val_accuracy * 100:.1f}%"
            f" | Val F1 {val_f1:.3f}"
            f" | LR {current_lr:.6f}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                "  Best model saved:",
                MODEL_PATH.name,
            )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best validation F1: "
        f"{best_val_f1:.3f}"
    )

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    (
        test_loss,
        test_accuracy,
        test_f1,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFINAL TEST")

    print(
        f"Loss: {test_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_accuracy * 100:.1f}%"
    )

    print(
        f"Macro F1: "
        f"{test_f1:.3f}"
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=STAGE_NAMES,
            digits=3,
            zero_division=0,
        )
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print(
        "\nRows = TRUE"
    )

    print(
        "Columns = PREDICTED\n"
    )

    print(
        "        Wake     N1      N2      N3     REM"
    )

    for name, row in zip(
        STAGE_NAMES,
        cm,
    ):

        values = " ".join(
            f"{value:7d}"
            for value in row
        )

        print(
            f"{name:5s} {values}"
        )

    print(
        "\nModel saved to:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()