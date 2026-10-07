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
- Policy uses the existing relaxation score and stage estimates. Carrier 200 Hz;
  beat 6–10 Hz, at most 0.5 Hz change every 30 seconds; master 0–0.10,
  at most 0.01 change per update; tone 0.01, wave 0.05, rain/birds zero.
- At two qualifying updates spaced at least 30 seconds apart with total
  N2/N3/REM probability >=0.8, target master decreases toward 0.03. This threshold
  is an engineering heuristic, not calibrated certainty or verified sleep.
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
