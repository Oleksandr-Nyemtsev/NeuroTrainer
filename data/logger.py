import csv
import os
from datetime import datetime


class SessionLogger:

    def __init__(self, filepath):

        self.filepath = filepath

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True
        )

        self.file = open(
            filepath,
            mode="w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "delta",
            "theta",
            "alpha",
            "beta",
            "gamma",
            "raw_score",
            "smoothed_score",
            "beat_frequency",
            "carrier_frequency"
        ])


    def log(
        self,
        state,
        raw_score,
        smoothed_score,
        action
    ):

        self.writer.writerow([
            datetime.now().isoformat(),
            state["Delta"],
            state["Theta"],
            state["Alpha"],
            state["Beta"],
            state["Gamma"],
            raw_score,
            smoothed_score,
            action["beat_frequency"],
            action["carrier_frequency"]
        ])

        self.file.flush()


    def close(self):

        self.file.close()