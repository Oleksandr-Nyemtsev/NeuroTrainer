from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.y = torch.from_numpy(
            y.astype(np.int64)
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index],
        )


def find_recordings(data_dir):

    data_dir = Path(data_dir)

    files = sorted(
        data_dir.glob("*.npz")
    )

    recordings = []

    for file in files:

        with np.load(file) as data:

            subject_id = int(
                data["subject_id"]
            )

            recording_id = int(
                data["recording_id"]
            )

        recordings.append(
            {
                "file": file,
                "subject_id": subject_id,
                "recording_id": recording_id,
            }
        )

    return recordings


def split_subjects(
    recordings,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
):

    subject_ids = sorted(
        set(
            item["subject_id"]
            for item in recordings
        )
    )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(
        subject_ids
    )

    total_subjects = len(subject_ids)

    train_end = int(
        total_subjects * train_ratio
    )

    val_end = int(
        total_subjects
        * (train_ratio + val_ratio)
    )

    train_subjects = set(
        subject_ids[:train_end]
    )

    val_subjects = set(
        subject_ids[train_end:val_end]
    )

    test_subjects = set(
        subject_ids[val_end:]
    )

    return (
        train_subjects,
        val_subjects,
        test_subjects,
    )


def load_subject_data(
    recordings,
    subject_ids,
):

    X_parts = []
    y_parts = []

    selected_files = [
        item
        for item in recordings
        if item["subject_id"] in subject_ids
    ]

    for index, item in enumerate(
        selected_files,
        start=1,
    ):

        file = item["file"]

        print(
            f"[{index}/{len(selected_files)}] "
            f"Loading {file.name}"
        )

        with np.load(file) as data:

            X_parts.append(
                data["X"]
            )

            y_parts.append(
                data["y"]
            )

    if len(X_parts) == 0:

        raise ValueError(
            "No recordings found for selected subjects."
        )

    X = np.concatenate(
        X_parts,
        axis=0,
    )

    y = np.concatenate(
        y_parts,
        axis=0,
    )

    return X, y