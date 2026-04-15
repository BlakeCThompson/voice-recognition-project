"""Data preparation script for creating train/val/test splits."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_splitter import (
    discover_characters,
    create_splits,
    save_splits,
    print_split_statistics
)
from src.data.preprocessing import validate_audio_file
from src.utils.config import load_config


def validate_dataset(character_files: dict) -> dict:
    """
    Validate audio files in dataset.

    Args:
        character_files: Dictionary of character names to file paths

    Returns:
        Dictionary with valid files only
    """
    print("\nValidating audio files...")

    valid_character_files = {}
    total_files = 0
    valid_files = 0
    invalid_files = 0

    for character, files in character_files.items():
        valid_files_for_char = []

        for file_path in files:
            total_files += 1
            is_valid, error_msg = validate_audio_file(file_path)

            if is_valid:
                valid_files_for_char.append(file_path)
                valid_files += 1
            else:
                invalid_files += 1
                print(f"  ⚠ Invalid: {Path(file_path).name} - {error_msg}")

        if valid_files_for_char:
            valid_character_files[character] = valid_files_for_char

    print(f"\nValidation complete:")
    print(f"  Total files: {total_files}")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {invalid_files}")

    return valid_character_files


def main():
    parser = argparse.ArgumentParser(description="Prepare audiobook character voice dataset")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--characters_dir',
        type=str,
        default=None,
        help='Path to characters directory (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for splits (overrides config)'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip audio file validation'
    )
    parser.add_argument(
        '--include_characters',
        type=str,
        nargs='+',
        default=None,
        metavar='CHARACTER',
        help='Only include these characters (space-separated). Overrides config.'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Resolve include_characters: CLI overrides config
    include_characters = args.include_characters or config.data.include_characters or None

    # Derive run name and output directory from character set
    if include_characters:
        run_name = "_".join(sorted(include_characters))
        default_output = str(Path(config.paths.data_splits_dir) / run_name)
    else:
        run_name = None
        default_output = config.paths.data_splits_dir

    # Get directories
    characters_dir = args.characters_dir or config.data.characters_dir
    output_dir = args.output_dir or default_output

    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"\nCharacters directory: {characters_dir}")
    print(f"Output directory: {output_dir}")
    if run_name:
        print(f"Run name: {run_name}")

    # Discover character audio files
    print("\nDiscovering character audio files...")
    character_files = discover_characters(characters_dir, include_characters=include_characters)

    if not character_files:
        print("\n❌ Error: No character directories found!")
        print(f"Expected structure:")
        print(f"  {characters_dir}/")
        print(f"    character1/")
        print(f"      clip1.wav")
        print(f"      clip2.wav")
        print(f"    character2/")
        print(f"      clip1.wav")
        sys.exit(1)

    print(f"\nFound {len(character_files)} characters:")
    for character, files in sorted(character_files.items()):
        print(f"  {character}: {len(files)} files")

    # Check minimum clips per character
    min_clips = config.data.min_clips_per_character
    insufficient_characters = [
        (char, len(files)) for char, files in character_files.items()
        if len(files) < min_clips
    ]

    if insufficient_characters:
        print(f"\n⚠ Warning: Some characters have fewer than {min_clips} clips:")
        for char, count in insufficient_characters:
            print(f"  {char}: {count} clips")

    # Validate audio files
    if not args.skip_validation:
        character_files = validate_dataset(character_files)

        if not character_files:
            print("\n❌ Error: No valid audio files found!")
            sys.exit(1)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_dict, val_dict, test_dict = create_splits(
        character_files,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        random_seed=config.data.random_seed
    )

    # Print statistics
    print_split_statistics(train_dict, val_dict, test_dict)

    # Save splits
    save_splits(train_dict, val_dict, test_dict, output_dir)

    print("\n✓ Data preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review splits in {output_dir}")
    if include_characters:
        chars_arg = " ".join(include_characters)
        print(f"  2. Train model: python scripts/train.py --include_characters {chars_arg}")
    else:
        print(f"  2. Train model: python scripts/train.py")


if __name__ == "__main__":
    main()
