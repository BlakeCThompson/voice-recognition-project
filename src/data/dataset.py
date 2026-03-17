"""PyTorch Dataset for character voice classification."""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable
from pathlib import Path

from .preprocessing import preprocess_audio
from .augmentation import AudioAugmenter


class CharacterVoiceDataset(Dataset):
    """Dataset for loading and preprocessing character voice audio."""

    def __init__(
        self,
        data_dict: Dict,
        sample_rate: int = 16000,
        duration: float = 8.0,
        target_db: float = -3.0,
        silence_threshold_db: float = 20.0,
        trim_silence: bool = True,
        augmenter: Optional[AudioAugmenter] = None,
        pad_mode: str = "random"
    ):
        """
        Initialize dataset.

        Args:
            data_dict: Dictionary containing 'data' (list of samples) and 'character_to_idx' mapping
            sample_rate: Target sample rate
            duration: Target duration in seconds
            target_db: Target amplitude in dB
            silence_threshold_db: Silence threshold for trimming
            trim_silence: Whether to trim silence
            augmenter: Audio augmenter instance (optional)
            pad_mode: Padding mode ('random', 'center', 'start', 'end')
        """
        self.data = data_dict['data']
        self.character_to_idx = data_dict['character_to_idx']
        self.idx_to_character = data_dict['idx_to_character']
        self.num_characters = data_dict['num_characters']

        self.sample_rate = sample_rate
        self.duration = duration
        self.target_db = target_db
        self.silence_threshold_db = silence_threshold_db
        self.trim_silence = trim_silence
        self.augmenter = augmenter
        self.pad_mode = pad_mode

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (waveform, label_idx)
        """
        sample = self.data[idx]
        audio_path = sample['audio_path']
        character_name = sample['character']
        label_idx = self.character_to_idx[character_name]

        # Preprocess audio
        try:
            waveform = preprocess_audio(
                audio_path=audio_path,
                target_sr=self.sample_rate,
                target_duration=self.duration,
                target_db=self.target_db,
                silence_threshold_db=self.silence_threshold_db,
                trim_silence_enabled=self.trim_silence,
                pad_mode=self.pad_mode
            )

            # Apply augmentation if available
            if self.augmenter is not None:
                waveform = self.augmenter(waveform)

            # Squeeze to (samples,) for wav2vec2
            waveform = waveform.squeeze(0)

            return waveform, label_idx

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros in case of error
            target_samples = int(self.sample_rate * self.duration)
            return torch.zeros(target_samples), label_idx


def create_dataloader(
    dataset: CharacterVoiceDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for dataset.

    Args:
        dataset: CharacterVoiceDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
