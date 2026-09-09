"""Waveform augmentations used by the selective-augmentation experiment."""

import random
from collections.abc import Collection

import numpy as np


def add_small_noise(signal: np.ndarray, noise_level: float = 0.003) -> np.ndarray:
    noise = np.random.randn(len(signal)).astype(np.float32)
    return signal + noise_level * noise


def apply_gain(signal: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    return signal * np.random.uniform(low, high)


def shift_waveform(signal: np.ndarray, max_shift_ratio: float = 0.03) -> np.ndarray:
    max_shift = int(len(signal) * max_shift_ratio)
    if max_shift <= 0:
        return signal
    return np.roll(signal, np.random.randint(-max_shift, max_shift + 1))


def maybe_augment(
    signal: np.ndarray,
    language: str,
    target_languages: Collection[str],
) -> np.ndarray:
    """Probabilistically augment selected languages and return float32 audio."""
    if language not in target_languages:
        return signal

    if random.random() < 0.35:
        signal = add_small_noise(signal, np.random.uniform(0.0015, 0.005))
    if random.random() < 0.30:
        signal = apply_gain(signal, 0.92, 1.08)
    if random.random() < 0.20:
        signal = shift_waveform(signal, 0.02)

    return np.clip(signal, -1.0, 1.0).astype(np.float32)
