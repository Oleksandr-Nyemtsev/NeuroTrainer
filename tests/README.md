# Automated regression tests

Run from the repository root in the activated neuro_env:

    python -B -m pytest -q -p no:cacheprovider

pytest.ini discovers all test_*.py files. Safe legacy software demos have assert tests.
Four data_heavy tests are collected but skipped by default: brain_to_audio,
real_brain_state, sleep_model_real, stimulation_policy. They require complete
local Sleep-EDF NPZ recordings and trained .pt weights. Use --run-data-heavy
only when explicitly intending to load those resources. Their imports are
always collected, so DLL/import failures are never hidden by these skips.
The manual adaptive-session main() starts audio; pytest tests policy only.
The regression suite uses synthetic arrays, temporary files, and fake audio/Muse.
It does not load model weights, connect Muse, train models, or download data.
The BrainFlow test calls native board metadata functions without connecting a board.
Three repaired legacy demos are smoke-tested with a fake audio engine.

Timing contract: time_to_n2_seconds = n2_onset - window_end.
ISRUC stable-N2 and Muse n2_onset annotations retain distinct label_definition values.
Legacy processed NPZ files without timestamps must be regenerated explicitly
before onset labels can be aligned; tests do not regenerate user data.

## Windows native DLL compatibility

Apply ../requirements-conda-windows.txt in neuro_env. NumPy's MKL backend can
load conda-forge llvm-openmp's libiomp5md.dll before torch. That shim lacks
_vcomp_barrier, _vcomp_for_static_end, _vcomp_for_static_simple_init_i8 and
_vcomp_fork, which torch 2.6.0 fbgemm.dll requires. Merely importing NumPy or
SciPy does not trigger this; numerical operations can. The pthreads OpenBLAS
backend avoids the incompatible runtime. Do not copy DLLs, preload torch in
conftest.py, or set KMP_DUPLICATE_LIB_OK to hide this conflict.

test_native_import_order.py exercises computation before torch in fresh
subprocesses and the reverse order. No CUDA operations, audio streams,
device connections, weights, or datasets are needed.

Full collection: python -m pytest --collect-only -q -o python_files="test_*.py" tests
All safe tests: python -m pytest -q tests
Original audit suite: python -m pytest -q -o python_files="test_audit_*.py" tests
