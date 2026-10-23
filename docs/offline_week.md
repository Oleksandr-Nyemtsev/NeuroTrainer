# Offline work: automatic audio collection

No new datasets or downloads are needed. Use the existing neuro_env. Keep the environment unchanged during roaming. Sound is generated locally. No sleep model runs in --audio-experiment mode.

From C:\Projects\NeuroTrainer:

    python run_sleep_observer.py --audio-experiment --audio --minutes 8 --condition awake-eyes-closed
    python analyze_session.py --latest
    python prepare_audio_trials.py --latest

Use awake-eyes-open for an open-eye session. Condition is self-report, not a classifier. Ctrl+C stops playback and recording. Check audio at a comfortable device volume; software levels are not calibrated sound pressure.

The exporter accepts only complete 60-second blocks followed by a valid transition. It rejects interruptions, mismatched sound settings and missing silent references. For sound, it compares the last 20 seconds against the preceding complete silence block's last 20 seconds. For silence drift, it compares seconds 10–30 with the last 20 seconds. At least five non-overlapping two-second windows are required per interval. Command timing allows 58–64 seconds of software exposure because device and command clocks differ; schedule boundaries must still span 60–64 seconds. The final unfinished block is excluded, even if the session ends normally. Transition-row EEG belongs to the previous command until the new command starts.

Outputs remain under ignored data/sessions/trials. Existing exports are never overwritten. Zero examples is an honest result, not a reason to relax signal gates. The interrupted, muted October 22 session cannot supply sound-response labels.

Exports contain observed relative band changes, provenance and observation IDs. They do not prove an audio effect, sleep, or recovery. They are not input to train_audio_response.py, whose older objective is manually rated comfort. Next work: define an EEG-response predictor with held-out whole sessions and comparison to silence drift/constant prediction. Do not split overlapping windows randomly between training and validation.

Git contains code/tests/docs, not private EEG recordings or weights. Keep a separate local/external-drive backup of data/sessions, models and the working environment. No hardware runs in pytest. Default tests skip data-heavy checks.
