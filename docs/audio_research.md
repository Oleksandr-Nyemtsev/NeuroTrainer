# Audio feedback and sleep research: shortlist

Checked 2026-10-07. This is a research inventory, not an added treatment protocol.
No third-party program was installed and no bulk EEG/audio dataset was downloaded.

## Relevant existing programs

| Source | What it does | Relevance to NeuroTrainer |
|---|---|---|
| [Dreamento](https://github.com/dreamento/dreamento) | Python toolbox for real-time sleep EEG, scoring, sensory stimulation and event annotations; oriented toward ZMax hardware; MIT license shown in repository | Reference for event markers and combined recording/stimulation workflows, not a drop-in Muse driver or a trained personal audio policy |
| [neurofeedback-muse](https://github.com/matpb/neurofeedback-muse) | Muse 2 Linux application; continuous tone pitch follows smoothed selected EEG measures; includes raw/feature recording and offline review | Example of signal-to-sound feedback. It does not establish that its sounds cause REM |
| [Manchester closed-loop MATLAB app](https://research.manchester.ac.uk/en/datasets/closed-loop-sound-stimulation-during-sleep-in-matlab/) | Detects a slow oscillation and schedules audio tones; GPL 3.0+ listing | Reference for timing architecture; no code copied into this project |

## Data and sounds

1. **Alpha closed-loop auditory stimulation (Hebron et al., 2024).**
   [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11185466/),
   [data/scripts, 141.8 MB, CC BY 4.0](https://zenodo.org/records/10834784).
   Uses brief pink-noise pulses tied to EEG phase. The archive is described as
   data/scripts for reproducing figures; do not assume it provides every raw
   signal needed for training. Pre-trough stimulation delayed N2+ onset relative
   to sham; this is evidence against simply treating more stimulation as better.
   First candidate for studying experimental design and data structure.

2. **REM closed-loop stimulation (Jaramillo et al., 2024).**
   [Study](https://pubmed.ncbi.nlm.nih.gov/39208441/),
   [methods](https://academic.oup.com/sleep/article/47/12/zsae193/7745355),
   [dataset](https://zenodo.org/records/10663994).
   Study involved 18 healthy young adults and stimulation during existing REM,
   not inducing REM from wakefulness. Methods use short pink-noise pulses and
   phase-locked alpha/theta stimulation. Dataset includes raw EEG and scoring;
   listed total is 170.9 GB, individual subject archives around 7 GB. License
   text was not exposed by the retrieved dataset page; verify metadata/license
   before reuse. Do not bulk-download for the current milestone.

3. **FSD50K / Freesound: audio assets, not EEG-response labels.**
   [Official dataset](https://zenodo.org/records/4060432).
   51,197 clips, 200 sound categories. Labels identify sound events, not which
   sound helps a person sleep. Individual licenses differ (CC0, CC BY, CC BY-NC,
   Sampling+); metadata maps each clip to its license and uploader. For an initial
   sound palette, consider a small individually checked CC0/CC BY selection with
   source, author and license recorded. Dataset-level conditions must also be
   checked before commercial use. There is no reason to download the full corpus
   just to obtain a few water/rain textures.

4. **OpenMIIR: EEG during music perception/imagination.**
   [Authors' repository](https://github.com/sstober/openmiir).
   Ten participants, twelve short musical excerpts, approximately 700 MB EEG per
   participant. PDDL dataset license stated by authors. Useful for music-response
   methods, but no sleep-induction outcome; underlying music redistribution rights
   need separate checking. Lower priority than a study with stimulation timing.

## Binaural beats

[Systematic review (PLOS ONE, 2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286023)
found inconsistent EEG entrainment evidence across 14 studies: five positive,
eight negative and one mixed. This does not support advertising a beat frequency
as a guaranteed route to a sleep stage. Binaural tones can be generated from
parameters; a collection of MP3 files is not required to implement them.

## Decision for this project

Keep the present audio policy unchanged. Before importing research data, inspect
the compact alpha-CLAS archive's event/subject schema and licensing. For assets,
start with a small provenance-tracked sound palette, not a claim of medically
"best" sounds. Train a personal selection policy only on suitable paired
observations; the sources above do not provide a ready-made universal policy.
Phase-locked lab stimulation requires latency validation and calibrated delivery;
our two-second feature loop is not equivalent to that apparatus.
