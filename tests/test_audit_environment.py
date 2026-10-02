import numpy as np
from brainflow.board_shim import BoardShim, BoardIds
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def test_brainflow_native_metadata_without_connecting():
    board = BoardIds.MUSE_2_BOARD.value
    channels = BoardShim.get_eeg_channels(board)
    assert len(channels) == 4
    assert BoardShim.get_sampling_rate(board) == 256
    assert BoardShim.get_timestamp_channel(board) < BoardShim.get_num_rows(board)


def test_riemannian_dependencies_on_small_synthetic_input():
    x = np.random.default_rng(42).normal(size=(5, 2, 100))
    cov = Covariances(estimator="scm").fit_transform(x)
    features = TangentSpace().fit_transform(cov)
    assert features.shape == (5, 3)
    assert np.isfinite(features).all()
