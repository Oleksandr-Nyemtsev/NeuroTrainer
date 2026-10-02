import math


def neutral_audio_action():
    # AudioEngine smooths the transition; do not choose a beat from invalid EEG.
    return {"master_volume": 0.0}


def choose_audio_action(score):
    if not math.isfinite(score):
        return neutral_audio_action()

    if score < 1.5:
        return {
            "beat_frequency": 6.0,
            "carrier_frequency": 200.0
        }

    elif score < 3.0:
        return {
            "beat_frequency": 8.0,
            "carrier_frequency": 200.0
        }

    else:
        return {
            "beat_frequency": 10.0,
            "carrier_frequency": 200.0
        }