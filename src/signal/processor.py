from src.signal.filters import bandpass_filter
from src.signal.artifacts import detect_artifact
from src.signal.features import analyze_channel


def process_channel(signal, fs=256):

    filtered = bandpass_filter(
        signal,
        fs=fs
    )

    artifact = detect_artifact(filtered)

    bands = analyze_channel(
        filtered,
        fs=fs
    )

    return {
        "filtered": filtered,
        "artifact": artifact,
        "bands": bands
    }