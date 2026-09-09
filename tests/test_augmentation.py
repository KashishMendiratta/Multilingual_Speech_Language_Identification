import numpy as np

from speech_lid.augmentation import apply_gain, maybe_augment, shift_waveform


def test_non_target_language_is_unchanged():
    signal = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    result = maybe_augment(signal, "english", {"hindi"})
    assert result is signal


def test_gain_preserves_shape(monkeypatch):
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 1.05)
    signal = np.ones(8, dtype=np.float32)
    result = apply_gain(signal)
    np.testing.assert_allclose(result, np.full(8, 1.05))


def test_short_waveform_is_not_shifted():
    signal = np.arange(3, dtype=np.float32)
    np.testing.assert_array_equal(shift_waveform(signal, 0.03), signal)
