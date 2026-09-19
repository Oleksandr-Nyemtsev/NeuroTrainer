import numpy as np


def build_state(results):

    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    state = {}

    for band in band_names:

        values = []

        for channel in results.values():
            values.append(
                channel["bands"][band]
            )

        state[band] = np.mean(values)

    return state