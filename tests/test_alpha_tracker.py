import numpy as np

from src.signal.alpha_tracker import AlphaTracker


FS = 256
DURATION = 20
TRUE_ALPHA = 10.0


t = np.arange(
    0,
    DURATION,
    1 / FS,
)


# Два EEG-канали з alpha 10 Hz
channel_1 = np.sin(
    2 * np.pi * TRUE_ALPHA * t
)

channel_2 = 0.8 * np.sin(
    2 * np.pi * TRUE_ALPHA * t + 0.15
)


# Трошки шуму, щоб тест був реальніший
rng = np.random.default_rng(42)

channel_1 += 0.15 * rng.normal(
    size=len(t)
)

channel_2 += 0.15 * rng.normal(
    size=len(t)
)


eeg = np.vstack(
    [
        channel_1,
        channel_2,
    ]
)


tracker = AlphaTracker(
    fs=FS
)


iaf = tracker.estimate_iaf(
    eeg
)

result = tracker.analyze(
    eeg
)


print(
    "True alpha frequency:",
    TRUE_ALPHA,
)

print(
    "Estimated IAF:",
    iaf,
)

print()

print(
    "Alpha amplitude:",
    result["alpha_amplitude"],
)

print(
    "Alpha phase:",
    result["alpha_phase"],
)

print(
    "Alpha stability:",
    result["alpha_stability"],
)
