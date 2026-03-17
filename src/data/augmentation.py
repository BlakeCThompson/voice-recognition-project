"""Data augmentation for audio."""

import torch
import numpy as np
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain
)
from typing import Optional, Dict, Any


class AudioAugmenter:
    """Audio augmentation pipeline using audiomentations."""

    def __init__(self, config: Dict[str, Any], sample_rate: int = 16000):
        """
        Initialize augmenter with configuration.

        Args:
            config: Augmentation configuration dictionary
            sample_rate: Audio sample rate
        """
        self.config = config
        self.sample_rate = sample_rate
        self.enabled = config.get('enabled', True)

        if self.enabled:
            self.augmentation_pipeline = self._build_pipeline()
        else:
            self.augmentation_pipeline = None

    def _build_pipeline(self) -> Compose:
        """Build augmentation pipeline from configuration."""
        augmentations = []

        # Gaussian noise
        if self.config.get('noise', {}).get('enabled', True):
            noise_config = self.config['noise']
            augmentations.append(
                AddGaussianNoise(
                    min_amplitude=noise_config.get('min_amplitude', 0.001),
                    max_amplitude=noise_config.get('max_amplitude', 0.015),
                    p=noise_config.get('probability', 0.7)
                )
            )

        # Time stretching
        if self.config.get('time_stretch', {}).get('enabled', True):
            stretch_config = self.config['time_stretch']
            augmentations.append(
                TimeStretch(
                    min_rate=stretch_config.get('min_rate', 0.9),
                    max_rate=stretch_config.get('max_rate', 1.1),
                    p=stretch_config.get('probability', 0.4),
                    leave_length_unchanged=False
                )
            )

        # Gain variation
        if self.config.get('gain', {}).get('enabled', True):
            gain_config = self.config['gain']
            augmentations.append(
                Gain(
                    min_gain_in_db=gain_config.get('min_gain_db', -6),
                    max_gain_in_db=gain_config.get('max_gain_db', 6),
                    p=gain_config.get('probability', 0.5)
                )
            )

        # Pitch shifting (be careful with this!)
        if self.config.get('pitch_shift', {}).get('enabled', True):
            pitch_config = self.config['pitch_shift']
            augmentations.append(
                PitchShift(
                    min_semitones=pitch_config.get('min_semitones', -1),
                    max_semitones=pitch_config.get('max_semitones', 1),
                    p=pitch_config.get('probability', 0.3)
                )
            )

        # Time shifting
        if self.config.get('time_shift', {}).get('enabled', True):
            shift_config = self.config['time_shift']
            augmentations.append(
                Shift(
                    min_fraction=shift_config.get('min_fraction', -0.2),
                    max_fraction=shift_config.get('max_fraction', 0.2),
                    p=shift_config.get('probability', 0.3),
                    rollover=False
                )
            )

        return Compose(augmentations)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to waveform.

        Args:
            waveform: Input waveform tensor (1, samples)

        Returns:
            Augmented waveform tensor (1, samples)
        """
        if not self.enabled or self.augmentation_pipeline is None:
            return waveform

        # Convert to numpy (audiomentations expects numpy)
        audio_numpy = waveform.squeeze(0).numpy()

        # Apply augmentations
        augmented_numpy = self.augmentation_pipeline(
            samples=audio_numpy,
            sample_rate=self.sample_rate
        )

        # Convert back to torch
        augmented_tensor = torch.from_numpy(augmented_numpy).unsqueeze(0)

        return augmented_tensor


def get_augmenter(config: Dict[str, Any], sample_rate: int = 16000) -> Optional[AudioAugmenter]:
    """
    Factory function to create augmenter.

    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate

    Returns:
        AudioAugmenter instance or None if disabled
    """
    if not config.get('enabled', True):
        return None

    return AudioAugmenter(config, sample_rate)
