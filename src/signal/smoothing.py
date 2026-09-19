from collections import deque
import numpy as np


class ScoreSmoother:

    def __init__(self, window_size=5):
        self.values = deque(maxlen=window_size)


    def update(self, score):
        self.values.append(score)

        return float(
            np.mean(self.values)
        )