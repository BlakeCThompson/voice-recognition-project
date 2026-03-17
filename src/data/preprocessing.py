"""Audio preprocessing utilities."""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform and sample rate.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (will resample if needed)

    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = torchaudio.load(audio_path)

    # Resample if necessary
    if sr != sample_rate:
        waveform = resample_audio(waveform, sr, sample_rate)
        sr = sample_rate

    return waveform, sr


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Audio waveform tensor
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert stereo audio to mono.

    Args:
        waveform: Audio waveform tensor (channels, samples)

    Returns:
        Mono waveform (1, samples)
    """
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = 20.0
) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate in Hz
        threshold_db: Silence threshold in dB

    Returns:
        Trimmed waveform
    """
    # Compute energy
    energy = waveform.pow(2).mean(dim=0)

    # Find non-silent regions
    threshold = 10 ** (-threshold_db / 10)
    mask = energy > threshold

    if mask.sum() == 0:
        # All silent, return original
        return waveform

    # Find start and end indices
    indices = torch.where(mask)[0]
    start_idx = indices[0].item()
    end_idx = indices[-1].item() + 1

    return waveform[:, start_idx:end_idx]


def normalize_amplitude(waveform: torch.Tensor, target_db: float = -3.0) -> torch.Tensor:
    """
    Normalize audio amplitude to target dB level.

    Args:
        waveform: Audio waveform tensor
        target_db: Target dB level

    Returns:
        Normalized waveform
    """
    # Compute current peak amplitude in dB
    max_amp = torch.abs(waveform).max()

    if max_amp == 0:
        return waveform

    current_db = 20 * torch.log10(max_amp)

    # Compute gain
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized = waveform * gain_linear

    # Prevent clipping
    max_val = torch.abs(normalized).max()
    if max_val > 1.0:
        normalized = normalized / max_val * 0.99

    return normalized


def pad_or_trim(
    waveform: torch.Tensor,
    target_length: int,
    mode: str = "random"
) -> torch.Tensor:
    """
    Pad or trim waveform to target length.

    Args:
        waveform: Audio waveform tensor (channels, samples)
        target_length: Target number of samples
        mode: Padding/trimming mode ('random', 'center', 'start', 'end')

    Returns:
        Padded or trimmed waveform
    """
    current_length = waveform.shape[1]

    if current_length == target_length:
        return waveform

    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length

        if mode == "random":
            # Random padding position
            pad_left = torch.randint(0, padding + 1, (1,)).item()
            pad_right = padding - pad_left
        elif mode == "center":
            pad_left = padding // 2
            pad_right = padding - pad_left
        elif mode == "start":
            pad_left = 0
            pad_right = padding
        elif mode == "end":
            pad_left = padding
            pad_right = 0
        else:
            raise ValueError(f"Invalid mode: {mode}")

        waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))

    else:
        # Trim
        excess = current_length - target_length

        if mode == "random":
            # Random crop position
            start_idx = torch.randint(0, excess + 1, (1,)).item()
        elif mode == "center":
            start_idx = excess // 2
        elif mode == "start":
            start_idx = 0
        elif mode == "end":
            start_idx = excess
        else:
            raise ValueError(f"Invalid mode: {mode}")

        waveform = waveform[:, start_idx:start_idx + target_length]

    return waveform


def preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    target_duration: float = 8.0,
    target_db: float = -3.0,
    silence_threshold_db: float = 20.0,
    trim_silence_enabled: bool = True,
    pad_mode: str = "random"
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for audio.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        target_duration: Target duration in seconds
        target_db: Target amplitude in dB
        silence_threshold_db: Silence threshold for trimming
        trim_silence_enabled: Whether to trim silence
        pad_mode: Padding mode for pad_or_trim

    Returns:
        Preprocessed audio tensor (1, samples)
    """
    # Load audio
    waveform, sr = load_audio(audio_path, target_sr)

    # Convert to mono
    waveform = convert_to_mono(waveform)

    # Trim silence
    if trim_silence_enabled:
        waveform = trim_silence(waveform, sr, silence_threshold_db)

    # Normalize amplitude
    waveform = normalize_amplitude(waveform, target_db)

    # Pad or trim to fixed length
    target_samples = int(target_sr * target_duration)
    waveform = pad_or_trim(waveform, target_samples, mode=pad_mode)

    # Validate
    assert waveform.shape[1] == target_samples, f"Expected {target_samples} samples, got {waveform.shape[1]}"
    assert not torch.isnan(waveform).any(), "NaN values in waveform"
    assert not torch.isinf(waveform).any(), "Inf values in waveform"

    return waveform


def validate_audio_file(audio_path: str, min_duration: float = 3.0) -> Tuple[bool, str]:
    """
    Validate audio file.

    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file exists
        if not Path(audio_path).exists():
            return False, "File does not exist"

        # Check file extension
        if not audio_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            return False, "Unsupported audio format (use WAV, MP3, FLAC, or OGG)"

        # Try to load
        waveform, sr = torchaudio.load(audio_path)

        # Check duration
        duration = waveform.shape[1] / sr
        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s < {min_duration:.1f}s)"

        # Check for silent audio
        if torch.abs(waveform).max() < 1e-6:
            return False, "Audio appears to be silent"

        return True, ""

    except Exception as e:
        return False, f"Error loading audio: {str(e)}"
