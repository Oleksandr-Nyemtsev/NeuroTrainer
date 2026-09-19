from pathlib import Path

import numpy as np
import torch

from src.models.sleep_cnn import SleepCNN


SLEEP_LABELS = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]


class SleepModel:

    def __init__(
        self,
        model_path,
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.model = SleepCNN()

        model_path = Path(
            model_path
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()


    def predict(
        self,
        eeg,
    ):
        """
        eeg expected shape:
        2 x 3000
        """

        eeg = np.asarray(
            eeg,
            dtype=np.float32,
        )

        if eeg.shape != (2, 3000):

            raise ValueError(
                f"Expected EEG shape (2, 3000), "
                f"got {eeg.shape}"
            )


        tensor = torch.from_numpy(
            eeg
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )


        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]


        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )


        return {
            label: float(prob)
            for label, prob
            in zip(
                SLEEP_LABELS,
                probabilities,
            )
        }