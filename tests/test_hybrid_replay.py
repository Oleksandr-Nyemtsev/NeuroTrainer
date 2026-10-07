import numpy as np
import pytest
import torch

from src.models.hybrid_crg_sleep import HybridCRGSleep
from src.models.hybrid_sleep_model import HybridSleepModel, LABELS
from replay_boas import load_recording, metrics


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "hybrid.pt"
    torch.save({"model_state_dict": HybridCRGSleep().state_dict(), "class_names": list(LABELS)}, path)
    return path


def normalized():
    x = np.random.default_rng(42).normal(size=(2, 2, 3000)).astype(np.float32)
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def test_loader_matches_direct_model_and_single_prediction(checkpoint):
    model = HybridSleepModel(checkpoint, "cpu")
    x = normalized()
    actual = model.predict_batch(x)
    with torch.inference_mode():
        expected = model.model(torch.from_numpy(x)).softmax(1).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)
    single = model.predict(x[0])
    assert tuple(single) == LABELS
    np.testing.assert_allclose(list(single.values()), actual[0], atol=1e-5)
    assert not model.model.training
    for bad in (x[:, :, :512], np.zeros_like(x), x * np.nan, x * 50):
        with pytest.raises(ValueError):
            model.predict_batch(bad)


def test_wrong_label_order_rejected(checkpoint):
    data = torch.load(checkpoint, weights_only=True)
    data["class_names"].reverse()
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="class_names"):
        HybridSleepModel(checkpoint, "cpu")


def test_recording_validation_and_time_gaps(tmp_path):
    path = tmp_path / "sub-1.npz"
    fields = dict(X=normalized(), y=np.array([0, 4]), onsets=np.array([0., 90.]),
                  sampling_frequency=100, subject_id="sub-1")
    np.savez(path, **fields)
    assert load_recording(path)[2].tolist() == [0, 90]
    for key, bad in (("sampling_frequency", 256), ("subject_id", "sub-2"),
                     ("y", np.array([0, 5])), ("onsets", np.array([0., 10.]))):
        np.savez(path, **{**fields, key: bad})
        with pytest.raises(ValueError):
            load_recording(path)


def test_metrics_include_missing_classes():
    accuracy, f1 = metrics(np.array([0, 0, 1]), np.array([0, 1, 1]))
    assert accuracy == pytest.approx(2/3)
    np.testing.assert_allclose(f1, [2/3, 2/3, 0, 0, 0])
