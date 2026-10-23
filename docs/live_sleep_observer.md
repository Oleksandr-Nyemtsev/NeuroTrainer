# Experimental live sleep observation

Run from NeuroTrainer in `neuro_env`:

```powershell
python run_sleep_observer.py --minutes 2
```

This connects to Muse and shows relative band powers every two seconds. After
30 seconds of continuous usable EEG, the BOAS Hybrid model estimates stages
from the last 30 seconds, refreshed every two seconds. These overlapping estimates
are correlated. AF7/AF8 are an experimental input mapping: BOAS performance does
not establish Muse accuracy. No audio device is opened by default.

Optional bounded audio (default duration 40 minutes, maximum 50):

```powershell
python run_sleep_observer.py --audio --minutes 40
```

Use `--audio-device INDEX` if the OS default output is not appropriate. Ctrl+C
stops the session and releases EEG/audio resources. Start with comfortable system
and headphone volume: software amplitudes are not calibrated acoustic levels.

## Behavior and limits

- Starts silent; missing/invalid/stale EEG mutes the target output and resets
  sleep context. AudioEngine smooths transitions, so target zero is not an
  instantaneous hardware mute.
- Keeps the existing two-second relative rhythm analysis. Existing `run_session.py`
  retains its previous behavior; this runner enables the optional observer/policy.
- AF7/AF8 are filtered 1–40 Hz, resampled 256→100 Hz, z-scored per channel and
  converted to float32 `(2,3000)`. No raw microvolt multiplication is needed.
- Usable EEG means the signal passed the engineering quality gates; it does
  not establish sleep or relaxation. `ready` means a prediction is available.
  Current BOAS-to-Muse predictions carry `stage_validated_for_input: false`;
  their stage/probabilities remain diagnostic model outputs.
- With `--audio`, policy uses the existing rhythm-based relaxation score. Carrier 200 Hz;
  beat 6–10 Hz, at most 0.5 Hz change every 30 seconds; master 0–0.10,
  at most 0.01 change per update; tone 0.01, wave 0.05, rain/birds zero.
- Sleep-specific attenuation requires explicit `stage_validated_for_input: true`.
  Missing/false validation resets sleep evidence immediately. Current Muse
  predictions cannot trigger this branch, even with N2/REM probability 1.
  The retained branch decreases target master toward 0.03 after two qualifying
  updates at least 30 seconds apart with N2/N3/REM probability >=0.8; this remains
  an engineering heuristic. No current Muse model is declared validated.
- No learned sound selection, causal inference, automatic search for stronger
  stimulation, guaranteed REM or demonstrated restoration in 40 minutes.
- Console diagnostics record rhythm measures, stage probabilities, window times
  and actions. They do not establish that an audio change caused an EEG change.

## Files

`src/core/sleep_observer.py`: rolling context and preprocessing.
`src/core/bounded_audio.py`: bounded rule-based policy.
`src/core/session.py`: optional integration alongside fast analysis.
`run_sleep_observer.py`: separate launch command, silent default.
`tests/test_sleep_observer.py`: preparation, timing, failures, audio limits and integration.

No new training is required. New datasets for learning an audio-response policy
would need stimulus timing/parameters and measured responses; additional sleep-stage
labels alone do not provide that supervision. Hardware operation must be checked
locally; automated tests simulate streams and do not play sound.


## Packet-aware timing check (2026-10-21)

Before starting the EEG stream, the console prints:
- Muse is about to start. Please get ready.
- Muse зараз почне запис. Будь ласка, підготуйтеся.

A five-second console countdown (5, 4, 3, 2, 1) follows before the EEG
stream starts. The session duration starts after this preparation. Ctrl+C
during the countdown disconnects Muse without starting the stream. No spoken audio.

Both the session and sleep observer use the same interval limit: 1.5/256 s
without packet evidence, or 2/256 s (7.8125 ms) with regular exposed packet IDs.
This is a bounded engineering tolerance, not proof of complete per-channel data.
Packet jumps/resets, malformed IDs, and unexpected completed packet lengths
reject the incoming chunk when a packet channel is available. Duplicate/backward
timestamps, intervals above the selected limit, and stale/future timestamps
remain rejected. EEG quality limits and original samples/timestamps are unchanged.
The original 5.859375 ms gap count remains in diagnostics for comparison;
`applied_gap_threshold_ms` and `applied_gap_count` show the actual timing rule.

Synthetic tests cover accumulation of a full sleep context with bounded timestamp
jitter and rejection/reset after a missing packet. A live Muse run is still needed.


## Optional raw diagnostic capture

Use one self-reported condition for the whole short session:

```powershell
python run_sleep_observer.py --minutes 2 --record-raw --condition awake-eyes-open
python analyze_session.py --latest
```

For a separate eyes-closed awake recording use `--condition awake-eyes-closed`.
These are user-reported conditions, not verified sleep labels. No automatic
condition switching or awake/sleep detection is implied. Audio remains off
without `--audio`. Capture is disabled unless `--record-raw` is supplied.

Next to `SESSION.jsonl`, `SESSION.raw/` holds `manifest.jsonl` and one numeric
`.npy` file per received BrainFlow read. Full board matrices are preserved before
filtering, trimming, timing checks and EEG rejection, including NaNs if received.
Channel indexes, units, sampling rate, BrainFlow version and checkpoint hash are
in the manifest metadata. Timestamps and packet IDs are rows of each matrix;
the board descriptor specifies which rows. Empty reads are retained too.
Read IDs match journal observation IDs during ordinary operation. A disk error
may leave an unmatched final read; do not silently treat it as a valid observation.
Writes fail rather than overwrite existing captures. Completed chunks survive
interruption; a hard process termination may leave a partial final file/manifest
line requiring inspection before replay. Normal Ctrl+C writes the ending status.

`src.data.raw_session.iter_raw_reads(directory)` yields saved reads in order
without accessing Muse or audio. This is a reader, not a completed replay runner:
original timestamps require an offline clock if replaying the live loop, whose
wall-clock staleness checks must not be used unchanged on historical data.

All captures remain under the ignored `data/sessions/` directory. They contain
personal EEG. No uploading or Git publication is performed by this feature.
