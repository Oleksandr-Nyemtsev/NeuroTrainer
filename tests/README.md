# Automated regression tests

Run from the repository root in the activated neuro_env:

    python -B -m pytest -q -p no:cacheprovider

pytest.ini discovers test_audit_*.py only. Other test_*.py files are retained
as manual demonstrations; some load local datasets or start audio/hardware.
The regression suite uses synthetic arrays, temporary files, and fake audio/Muse.
It does not load model weights, connect Muse, train models, or download data.
The BrainFlow test calls native board metadata functions without connecting a board.
Three repaired legacy demos are smoke-tested with a fake audio engine.

Timing contract: time_to_n2_seconds = n2_onset - window_end.
ISRUC stable-N2 and Muse n2_onset annotations retain distinct label_definition values.
Legacy processed NPZ files without timestamps must be regenerated explicitly
before onset labels can be aligned; tests do not regenerate user data.
