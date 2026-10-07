# Session analysis and report

Run from the NeuroTrainer directory in `neuro_env`:

```powershell
python run_sleep_observer.py --minutes 2
python analyze_session.py --latest
```

The first command records a silent Muse session. The second needs no device,
audio or GPU: it selects the latest journal and writes a Markdown report and
machine-readable JSON under `data/sessions/reports/`. These personal files are
ignored by Git. To choose a particular session:

```powershell
python analyze_session.py "data/sessions/YOUR_SESSION.jsonl"
```

Each analysis creates a new report directory. Source journals are read-only.
If the source ended abruptly, a partial final JSON line may be skipped with a
warning; corrupted interior lines, duplicate IDs and wrong schemas are rejected.

## What the report answers

- Was EEG recorded, and how many observations passed the signal gate?
- Why were observations rejected?
- Was audio enabled or was this a silent run?
- What software audio targets were active before each subsequent observation?
- How did relative band powers differ from the pre-command observation?

Before/after links are independently checked against timestamps and quality
segments. Windows crossing a command or a data-quality interruption are excluded.
Overlapping windows are not counted twice within a comparison. A band difference
requires at least three usable non-overlapping fast windows. Otherwise the report
says `insufficient_data`; this minimum is an engineering reporting rule, not a
statistical significance threshold. Differences are in percentage points.

Stage prediction counts in JSON count model outputs, not time asleep. There are
no ground-truth sleep labels for live sessions. Sound parameters are commands,
not measured acoustic delivery; smooth transitions can continue after commands.
The report does not rank sounds, assign rewards, or claim that sound caused a
change. Silent recordings remain labelled silent even when a policy proposed audio.

## Completion criteria for these three blocks

1. **Journal:** unique local files, flush after records, precise run outcome,
   model provenance, separate requested/effective actions, error/interrupt closure.
2. **Analysis:** independently validated before/after links, quality interruption
   handling, no overlap double counting, explicit missing/insufficient data.
3. **Report:** reproducible JSON plus readable Markdown, source checksum, safe
   handling of incomplete recordings, documented command and automated tests.

Personal sound learning is a subsequent block. These blocks provide inspectable
data for it; they do not make the whole research project clinically validated.
