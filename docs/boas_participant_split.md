# BOAS participant-independent training

`sub-*` identifies a BOAS recording, not a unique person. The processed NPZ
`subject_id` stores that recording ID. Resolve it through
`data/external/boas/participants.tsv` and group by `pid` before splitting.
All recordings from one person now stay in the same train/validation/test group.
The deterministic split uses seed 42 and approximately 70/15/15% of eligible people.

## Run

In the `neuro_env` environment, from the repository root:

```powershell
python train_boas_hybrid.py --prepare-only
python train_boas_hybrid.py
```

Preparation only writes the split and run configuration; it does not train.
Training starts from `models/isruc_training/hybrid_isruc_best.pt`, not from the old
BOAS checkpoint. Defaults remain 12 epochs, batch size 4, learning rate 0.0001,
early stopping after 4 epochs without validation macro-F1 improvement.
The best validation checkpoint is used for one final held-out test evaluation.

New results go to `models/boas_training_pid/`. Existing training results cause
an error; select a new `--output-dir` for another run. An unchanged prepare-only
configuration can be reused. The saved checkpoint includes configuration and
participant assignments. Parent-checkpoint and participants-table hashes are saved.

Old `models/boas_training/` results and all source/processed data remain unchanged.
Their metrics must not be described as independent-person performance because
the old recording-level split shared people between groups. New metrics are only
available after retraining. This change does not validate live Muse inference or
the effects of adaptive audio on sleep.

## GitHub

Track the training script, split helper, tests and this document in Git.
Model weights, datasets and generated run outputs remain covered by existing
repository ignore rules; a Git push does not back them up.
