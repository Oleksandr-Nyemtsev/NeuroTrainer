import numpy as np
import pytest
from src.core.packet_diagnostics import MusePacketDiagnostics


def test_repeated_ids_and_read_boundary_are_normal():
    tracker = MusePacketDiagnostics()
    tracker.update([100] * 5)
    d = tracker.update([100] * 7 + [101] * 12 + [102] * 4)
    assert d['consecutive_packet_steps'] == 2
    assert d['completed_packet_runs'] == 1
    assert d['unexpected_run_lengths'] == 0
    assert d['forward_jump_count'] == 0
    assert d['open_run_samples'] == 4


def test_jump_across_reads_and_wrap():
    tracker = MusePacketDiagnostics()
    tracker.update([65535] * 12)
    d = tracker.update([0] * 12 + [3] * 12)
    assert d['wrap_count'] == 1
    assert d['forward_jump_count'] == 1
    assert d['suspected_missing_packets'] == 2
    assert d['jump_examples'] == [{'from': 0, 'to': 3, 'modulo_step': 3}]


def test_reset_and_wrong_run_length_not_reported_as_massive_loss():
    tracker = MusePacketDiagnostics()
    tracker.update([100] * 12 + [101] * 3)
    d = tracker.update([102] * 12 + [1] * 12)
    assert d['unexpected_run_lengths'] == 1
    assert d['backward_or_reset_count'] == 1
    assert d['suspected_missing_packets'] == 0


@pytest.mark.parametrize('values', [[np.nan], [-1], [65536], [1.5], []])
def test_invalid_values_reset_tracker(values):
    tracker = MusePacketDiagnostics()
    tracker.update([100] * 12)
    assert not tracker.update(values)['available']
    assert tracker.last_id is None


def test_stuck_counter_is_visible():
    d = MusePacketDiagnostics().update([2] * 24)
    assert d['open_run_exceeds_expected']
