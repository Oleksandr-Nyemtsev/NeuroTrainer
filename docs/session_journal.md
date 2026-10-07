# Personal state/action journal

`python run_sleep_observer.py --minutes 2` now automatically creates a unique
`data/sessions/<UTC timestamp>_<id>.jsonl` file. Each line is a JSON record,
flushed after writing. Existing sessions are never overwritten. This directory
is excluded from Git. No raw EEG is saved by this journal.

The first record identifies the checkpoint hash, policy version, channels,
sampling rate, audio mode and planned duration. Each observation includes
EEG window timestamps, available band powers, signal quality, sleep estimates,
the requested audio action and software command timestamps. The last record
contains the observation count and status (`completed`, `connection_failed`, `stream_start_failed`, `interrupted`,
`error`). Historical journals may use `returned`, which does not prove successful
connection. `completed` indicates that the timed loop finished, not that EEG was
usable or that sleep occurred.
An abruptly terminated process may have no session-end record.

`requested_action` is a proposal. For silent runs `effective_targets` always
has zero master volume, even when the policy proposes sound. For audio-enabled
runs, targets are merged with previous commands because muting changes only
master volume. Current software audio parameters are captured separately;
they are smoothed and do not prove actual sound output or calibrated loudness.

`before_observation_id` links a target change to the observation used to choose
it. `fast_window_after_action_id` and `sleep_window_after_action_id` are only set
when the entire corresponding window follows the prior target command in the
same uninterrupted quality segment. Windows straddling a change are not linked.
Recovery after rejected data begins a new segment, including for unchanged sound.
No reward, causal effect, comfort rating or verified sleep label is inferred.

This is the timeline needed for subsequent analysis of personal responses. It
does not yet train a sound-selection policy. The existing audio policy and signal
thresholds are unchanged. No Muse or sound is started by automated tests.

Analyze a saved journal with `python analyze_session.py --latest`; see [analysis and report](session_analysis.md).
