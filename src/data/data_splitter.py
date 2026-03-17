"""Data splitting utilities for creating train/val/test splits."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split


def discover_characters(characters_dir: str) -> Dict[str, List[str]]:
    """
    Discover character audio files from directory structure.

    Expected structure:
        characters/
            character1/
                clip1.wav
                clip2.wav
            character2/
                clip1.wav

    Args:
        characters_dir: Path to characters directory

    Returns:
        Dictionary mapping character names to list of audio file paths
    """
    characters_dir = Path(characters_dir)

    if not characters_dir.exists():
        raise FileNotFoundError(f"Characters directory not found: {characters_dir}")

    character_files = defaultdict(list)

    # Scan subdirectories
    for character_dir in sorted(characters_dir.iterdir()):
        if not character_dir.is_dir():
            continue

        character_name = character_dir.name

        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(character_dir.glob(f'*{ext}'))
            audio_files.extend(character_dir.glob(f'*{ext.upper()}'))

        # Store absolute paths as strings
        character_files[character_name] = [str(f.absolute()) for f in sorted(audio_files)]

    return dict(character_files)


def create_splits(
    character_files: Dict[str, List[str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Create stratified train/validation/test splits.

    Args:
        character_files: Dictionary mapping character names to file paths
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(random_seed)

    train_data = []
    val_data = []
    test_data = []

    # Split each character's files
    for character_name, file_paths in character_files.items():
        num_files = len(file_paths)

        if num_files < 3:
            print(f"Warning: Character '{character_name}' has only {num_files} files. "
                  f"Need at least 3 for splits.")
            continue

        # Shuffle files
        shuffled_files = file_paths.copy()
        random.shuffle(shuffled_files)

        # Calculate split sizes
        train_size = int(num_files * train_ratio)
        val_size = int(num_files * val_ratio)
        # test_size is the remainder

        # Ensure at least 1 file in each split if possible
        if num_files >= 3:
            train_size = max(1, train_size)
            val_size = max(1, val_size)
            test_size = num_files - train_size - val_size
            test_size = max(1, test_size)

            # Adjust if needed
            if train_size + val_size + test_size != num_files:
                train_size = num_files - val_size - test_size

        # Split files
        train_files = shuffled_files[:train_size]
        val_files = shuffled_files[train_size:train_size + val_size]
        test_files = shuffled_files[train_size + val_size:]

        # Add to splits with character labels
        for fp in train_files:
            train_data.append({"audio_path": fp, "character": character_name})
        for fp in val_files:
            val_data.append({"audio_path": fp, "character": character_name})
        for fp in test_files:
            test_data.append({"audio_path": fp, "character": character_name})

    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Create character to index mapping
    character_names = sorted(character_files.keys())
    char_to_idx = {name: idx for idx, name in enumerate(character_names)}
    idx_to_char = {idx: name for name, idx in char_to_idx.items()}

    # Package into dictionaries
    train_dict = {
        "data": train_data,
        "character_to_idx": char_to_idx,
        "idx_to_character": idx_to_char,
        "num_characters": len(character_names)
    }

    val_dict = {
        "data": val_data,
        "character_to_idx": char_to_idx,
        "idx_to_character": idx_to_char,
        "num_characters": len(character_names)
    }

    test_dict = {
        "data": test_data,
        "character_to_idx": char_to_idx,
        "idx_to_character": idx_to_char,
        "num_characters": len(character_names)
    }

    return train_dict, val_dict, test_dict


def save_splits(
    train_dict: Dict,
    val_dict: Dict,
    test_dict: Dict,
    output_dir: str
):
    """
    Save data splits to JSON files.

    Args:
        train_dict: Training data dictionary
        val_dict: Validation data dictionary
        test_dict: Test data dictionary
        output_dir: Output directory for split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_dict, f, indent=2)

    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_dict, f, indent=2)

    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_dict, f, indent=2)

    print(f"Splits saved to {output_dir}")


def load_split(split_path: str) -> Dict:
    """
    Load a data split from JSON file.

    Args:
        split_path: Path to split JSON file

    Returns:
        Split dictionary
    """
    with open(split_path, 'r') as f:
        return json.load(f)


def print_split_statistics(train_dict: Dict, val_dict: Dict, test_dict: Dict):
    """
    Print statistics about the data splits.

    Args:
        train_dict: Training data dictionary
        val_dict: Validation data dictionary
        test_dict: Test data dictionary
    """
    print("\n" + "=" * 60)
    print("DATA SPLIT STATISTICS")
    print("=" * 60)

    # Overall statistics
    total_samples = len(train_dict['data']) + len(val_dict['data']) + len(test_dict['data'])
    print(f"\nTotal samples: {total_samples}")
    print(f"  Train: {len(train_dict['data'])} ({len(train_dict['data'])/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_dict['data'])} ({len(val_dict['data'])/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_dict['data'])} ({len(test_dict['data'])/total_samples*100:.1f}%)")

    # Per-character statistics
    print(f"\nNumber of characters: {train_dict['num_characters']}")
    print("\nPer-character breakdown:")
    print(f"{'Character':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)

    char_to_idx = train_dict['character_to_idx']

    for character in sorted(char_to_idx.keys()):
        train_count = sum(1 for item in train_dict['data'] if item['character'] == character)
        val_count = sum(1 for item in val_dict['data'] if item['character'] == character)
        test_count = sum(1 for item in test_dict['data'] if item['character'] == character)
        total_count = train_count + val_count + test_count

        print(f"{character:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")

    print("=" * 60 + "\n")
