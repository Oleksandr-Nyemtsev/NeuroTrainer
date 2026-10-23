# Personal EEG baseline — diagnostic only

The optional baseline compares measured theta/alpha/beta with this person's
previous awake recordings, separately for TP9, AF7, AF8 and TP10. It does not
train or calibrate a sleep classifier, change audio commands, or identify ADHD,
meditation, sleep, REM or recovery. No code/weights/thresholds from NeuroSkill
were copied. Its guided-condition concept informed the control protocol.

## Build from saved controls, without Muse

From NeuroTrainer in neuro_env (replace example session names with your local journals):

```powershell
python build_personal_baseline.py --eyes-open data/sessions/EYES_OPEN_SESSION.jsonl --eyes-closed data/sessions/EYES_CLOSED_SESSION.jsonl --output data/sessions/baselines/personal_v1.json
```

The command requires two completed, silent, correctly labelled journals with
completed sibling `.raw` captures. Conditions are user self-reports, not model
labels. Old sleep predictions are ignored. Original accepted fast windows must
match raw timestamps and samples exactly. Gaps, overlaps and bad channel quality
are excluded and counted. At least 20 non-overlapping windows (nominal 40 seconds)
per condition are required; this is an engineering minimum, not clinical validity.

The profile saves each condition separately, channel medians, quality/selection
counts, the processing method, and source journal/manifest/raw-chunk hashes.
It never replaces an existing output. Keep profiles under ignored `data/sessions`;
they contain personal measurements and must not be committed to GitHub.

## Optional live comparison (user starts hardware)

```powershell
python run_sleep_observer.py --minutes 2 --record-raw --personal-baseline data/sessions/baselines/personal_v1.json
```

Default reference is `awake-eyes-open`; select `--baseline-reference awake-eyes-closed`
to compare with the other saved condition. The condition label for a new recording
is independent: add `--condition awake-eyes-open` or `awake-eyes-closed` only if
that self-report genuinely describes the whole session. Audio stays off unless
explicitly enabled with the existing `--audio` option.

The `personal_baseline` field appears in the returned/printed observation and the
journal, with reference condition, profile SHA256 and per-channel/band changes:

```text
Alpha: {delta_pp: 5.0, power_ratio: 1.4}
```

`delta_pp=5` means five percentage points more relative alpha than that channel's
reference median. `power_ratio=1.4` means 1.4 times the reference absolute alpha
power. These numbers describe different properties; increased relative share can
coexist with decreased absolute power. Ratios are unavailable (`null`) when the
reference power is <=1e-6 µV² to avoid division by near-zero numerical noise.
This floor is a display rule, not a clinical threshold.

There is no learned "good" range and no combined state score. The profile never
updates itself during a session. It applies only to the selected person's setup;
changed contact, placement, electrical environment or day can change the values.
Do not interpret deviations as effects caused by sound or proof of a mental state.

## Exact processing and separation from the old score

Raw Muse 4x512 at 256 Hz, µV. Independently recheck all four channels with existing
MuseSignalQuality >=.75. Butterworth order4 1–40 Hz sosfiltfilt within each supplied
two-second window. Welch Hann nperseg256, overlap128, detrend constant, density.
Sum PSD x df: theta [4,8), alpha [8,13), beta [13,30); relative denominator [1,40).
Saved medians use only non-overlapping accepted windows; baseline and live
comparison use the exact same feature function. Unfiltered input units must be
preserved: never feed normalized model tensors to this path.

The pre-existing global `state`/relaxation score uses another method. It is left
unchanged, and must not be equated numerically with these per-channel measures.
Existing timestamp, latency, signal-quality and audio protections remain active.
On rejected/missing EEG the baseline field is not ready and contains no previous
channel values. A baseline-only numerical rejection is reported without substituting
an audio action. Processing time remains included in the existing latency gate.

Without `--personal-baseline`, no extra live feature calculation is performed.
The profile is validated before device connection, and bad versions/methods,
missing channels/conditions or nonfinite references fail explicitly.

## Checks

```powershell
python -m pytest -q tests/test_personal_baseline.py
python -m pytest -q
```

Tests use synthetic input/temp captures and simulated sessions, no EEG device,
audio output or model training. They check unchanged existing audio decisions,
same-signal reference comparison, scaling vs percentage, provenance, invalid
profiles, bad sources and fault handling. Source recordings used for this profile
are calibration/development data, not an independent test of future performance.
