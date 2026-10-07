# Offline BOAS model replay

From the NeuroTrainer directory in the `neuro_env` environment:

```powershell
python replay_boas.py
```

The default loads `models/boas_training_pid/hybrid_boas_best.pt` and evaluates
the first validation recording in its embedded participant split. No training,
device connection, audio playback, or output-file changes take place.

The console shows the first 20 epochs with their original onset times, true
stage, predicted stage and maximum model probability. All retained epochs in
the recording contribute to accuracy and five-class macro-F1. Missing classes
receive zero F1, matching training evaluation. These probabilities are not
calibrated confidence estimates. A single recording is not a new estimate of
whole-dataset performance and must not be used to tune against the held-out test.

Options: `--show 50`, `--device cpu`, `--recording sub-101`, `--batch-size 16`,
`--checkpoint PATH`, `--data-dir PATH`. Explicit relative paths resolve against
the current directory; defaults resolve against the script directory.

`HybridSleepModel.predict()` returns Wake/N1/N2/N3/REM probabilities in the
same dictionary interface consumed by `BrainStateEstimator`. The existing
SleepCNN loader and live session remain unchanged.

Input must be prepared BOAS data: HB_1/HB_2, 30-second epochs, bandpass 1–40 Hz,
resampled from 256 to 100 Hz, per-channel z-score, shape `(2, 3000)`. The loader
checks shape, finiteness and normalization; those checks cannot establish signal
origin or electrode equivalence. It does not accept raw four-channel Muse data.
Processed BOAS archives may omit rejected epochs; replay preserves time gaps.

This verifies saved-model inference on recordings. Live Muse buffering,
preprocessing, channel validation and connection to audio are separate work.
