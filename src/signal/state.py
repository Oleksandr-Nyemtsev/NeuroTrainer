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

        if not values or not np.isfinite(values).all():
            raise ValueError("Band powers must be finite and nonempty")
        state[band] = float(np.mean(values))

    return state