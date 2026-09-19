from src.signal.processor import process_channel


CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def process_eeg(eeg, fs=256):
    results = {}

    for i, channel_name in enumerate(CHANNEL_NAMES):
        results[channel_name] = process_channel(
            eeg[i],
            fs=fs
        )

    return results