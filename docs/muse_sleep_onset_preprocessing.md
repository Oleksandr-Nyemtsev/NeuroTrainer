# Muse Sleep-Onset preprocessing

This independent script prepares regression data only. It does not import model,
training, live Muse, BOAS, ISRUC or Sleep-EDF modules.

## Issues recorded before preprocessing

The dataset README says its supplied train/test split shares participants.
That split does not satisfy this task's subject isolation requirement. This
pipeline instead mirrors the existing `src/data/sleep_dataset.py` standard:
sorted unique subjects, NumPy default RNG with seed 42, shuffled once, 70/15/15
with floor cutoffs. The manifest includes every discovered subject, so subset
runs use the same assignments as a full run. For 203 subjects: 142/30/31.

The source README states that recording length reveals N2 timing because each
recording ends 300 seconds after N2. Full duration, N2 annotation, subject/session
identifiers, quality summaries and label metadata are provenance, not input
features. Consumers must feed EEG alone and must not infer labels from EOF.

The initial development scan (`muse_sleep_onset_audit_20261006_scan`) erroneously
included filenames in the subject list (743 entries). It is preserved but INVALID
and must not be used. Subject discovery was corrected to inspect directories only,
and an integration regression assertion now checks the exact subject universe.
The corrected scan is `muse_sleep_onset_audit_20261006_scan_v2`.

## Data contract

- Read BrainVision through MNE to apply header units/scaling; stored EEG is in volts.
- Native `raw.info['sfreq']`; no hardcoded sampling rate, resampling, filtering,
  normalization, artifact rejection or dtype downcast.
- Exactly ordered channels: TP9, AF7, AF8, TP10; missing/duplicate names are errors.
- Five-second windows, five-second steps, recording-relative times. Intervals
  are `[start_sample, end_sample)` and `[window_start, window_end)`.
- Keep only full windows with `window_end <= n2_onset`; discard a trailing partial
  window. Read EEG only up to the last retained end, never at/after N2.
- Target is `clip(n2_onset - window_end, 0, 600)` seconds. Labels may use the
  annotation; input transformations never use future EEG.
- Every recording archive has `X` shaped `(N, 4, samples_per_window)` in float64,
  `window_ids`, `channels`, `sampling_frequency`, `units`. `window_index` explicitly
  addresses `X[window_index]`; `window_id` must also match `window_ids[window_index]`.
- `windows.csv` and train/val/test CSVs include identifiers, timing, targets,
  shapes, split, archive-relative path and sample boundaries. No row-order key.
- Reopen saved archives and serialized CSV for finite, channel, shape, ID, timing,
  target and subject split verification before declaring success.

Inventory covers the entire input even in a subset run. It saves header/event
SHA256, EEG size/mtime, MNE/NumPy/Python versions (in processing report), and
per-recording native rates. Missing/invalid events, nonfinite EEG and invalid
channels are reported; any failure gives exit code 1 and an incomplete report.
No incomplete output directory is silently reused. Select a new path to retry.
The scan-only mode writes inventory and the split manifest; it does not read
EEG samples or produce windows.

Mixed sampling rates stop processing after writing inventory. The proposed
strategy is native per-recording archives and downstream grouping by frequency,
explicitly enabled with `--allow-mixed-sfreq` after reviewing inventory. This
never resamples. Rates for which five seconds is not an integral sample count
are rejected and require a separate timing strategy.

## Metrics

`muse_sleep_onset_metrics.py` exposes `mae_seconds` and `binned_mae_seconds`.
Bins use the true target: [0,40), [40,90), [90,300), [300,600]. Empty bins have
`count=0` and `mae_seconds=None`. Predictions are not clipped by metrics.
These utilities do not claim to implement weighted competition W-bMAE.

## Reproduction (PowerShell from C:\Projects\NeuroTrainer)

Use the existing neuro_env interpreter. Every output path must be new; `_rerun`
below avoids the initial outputs. For further repetitions choose another suffix.

```powershell
$py = 'C:\Users\user\anaconda3\envs\neuro_env\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py -m pytest tests\test_muse_sleep_onset.py -q -p no:cacheprovider
& $py prepare_muse_sleep_onset_all.py --output-dir data\processed\muse_sleep_onset_scan_rerun --scan-only
& $py prepare_muse_sleep_onset_all.py --output-dir data\processed\muse_sleep_onset_subject_rerun --subject sub-001
& $py prepare_muse_sleep_onset_all.py --output-dir data\processed\muse_sleep_onset_five_rerun --max-recordings 5
& $py prepare_muse_sleep_onset_all.py --output-dir data\processed\muse_sleep_onset_all_rerun
```

Tests use small synthetic BrainVision files and MNE RawArray objects. They do
not modify the real dataset. Any pytest temporary directory must be writable;
under a restricted sandbox use `--basetemp` with a new writable scratch path.
