# Automatic audio collection v1

Run from the repository with neuro_env active:

    python run_sleep_observer.py --audio-experiment --audio --minutes 8 --condition awake-eyes-closed
    python analyze_session.py --latest

Without --audio, the same schedule is a silent dry run, not an audio-response experiment.
Raw recording is enabled automatically. The sleep checkpoint is not loaded; no REM claim is produced.

After 60 seconds of uninterrupted usable EEG, software selects one of 6/8/10 Hz binaural beat differences (200 Hz carrier), in seeded shuffled order. These are experimental parameters, not proven sleep-induction frequencies. Each 60-second sound block is followed by 60 seconds of silence. Soundscape layers are disabled to isolate the chosen stimulus. Existing engine smoothing remains active; command timestamps are not measured acoustic onset. Volume is bounded to the existing candidate level .03, not a calibrated physical loudness.

A rejected or incomplete EEG observation mutes playback and starts a fresh silent reference. Interrupted blocks remain in the journal and must not be treated as completed trials. Consequently 8 minutes does not guarantee all candidates if signal quality is poor. Ctrl+C terminates the session and releases the devices.

Journal audio_experiment fields identify block, candidate, seed and mode. Existing analysis reports descriptive associations only. This implementation collects data automatically; it does not yet learn choices, establish causal effects, or validate sleep stages. Next: build complete-block before/after examples, compare with silent blocks and evaluate a response predictor on separate sessions. Neither rising alpha/theta nor network confidence is a REM label.
