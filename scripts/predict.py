"""Prediction script for character voice classification.

This is the main entry point for making predictions on new audio samples.

Usage:
    # Predict single file
    python scripts/predict.py --audio new_samples/unknown_clip.wav

    # Predict all files in new_samples/
    python scripts/predict.py --batch

    # With confidence threshold warning
    python scripts/predict.py --audio clip.wav --threshold 0.6

    # Show only top K predictions
    python scripts/predict.py --audio clip.wav --top_k 3
"""

import argparse
import sys
import torch
from pathlib import Path
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.preprocessing import preprocess_audio, validate_audio_file
from src.models.wav2vec2_classifier import CharacterVoiceClassifier
from src.utils.visualization import create_prediction_bar


def load_model_and_metadata(config):
    """Load trained model and metadata."""
    # Model path
    model_path = Path(config.paths.models_dir) / "best_model.pt"

    if not model_path.exists():
        print(f"\n{Fore.RED}❌ Error: Model not found: {model_path}")
        print(f"{Fore.YELLOW}Please train a model first: python scripts/train.py")
        sys.exit(1)

    # Metadata path
    metadata_path = Path(config.paths.models_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"\n{Fore.RED}❌ Error: Metadata not found: {metadata_path}")
        sys.exit(1)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda_if_available else 'cpu')

    # Create model
    model = CharacterVoiceClassifier(
        num_characters=metadata['num_characters'],
        model_name=metadata['model_config']['name'],
        freeze_layers=metadata['model_config']['freeze_layers'],
        dropout=metadata['model_config']['dropout'],
        hidden_dims=metadata['model_config']['hidden_dims']
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, metadata, device


def predict_audio(
    model,
    audio_path: str,
    metadata: dict,
    config,
    device: torch.device
) -> dict:
    """
    Make prediction on audio file.

    Args:
        model: Trained model
        audio_path: Path to audio file
        metadata: Model metadata
        config: Configuration
        device: Device to run on

    Returns:
        Dictionary with predictions and probabilities
    """
    # Preprocess audio
    waveform = preprocess_audio(
        audio_path=audio_path,
        target_sr=config.data.sample_rate,
        target_duration=config.data.duration,
        target_db=config.preprocessing.target_db,
        silence_threshold_db=config.preprocessing.silence_threshold_db,
        trim_silence_enabled=config.preprocessing.trim_silence,
        pad_mode="center"  # Use center for inference
    )

    # Add batch dimension and move to device
    waveform = waveform.squeeze(0).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        logits = model(waveform)
        probabilities = torch.softmax(logits, dim=1)

    # Get results
    probs = probabilities.squeeze(0).cpu().numpy()
    pred_idx = probs.argmax()

    # Create character probabilities dictionary
    character_probs = {}
    for idx, prob in enumerate(probs):
        character_name = metadata['idx_to_character'][str(idx)]
        character_probs[character_name] = float(prob)

    predicted_character = metadata['idx_to_character'][str(pred_idx)]
    confidence = float(probs[pred_idx])

    return {
        'predicted_character': predicted_character,
        'confidence': confidence,
        'all_probabilities': character_probs
    }


def print_prediction_result(
    audio_filename: str,
    result: dict,
    threshold: float = 0.5,
    top_k: int = None
):
    """
    Print prediction result in a nice format.

    Args:
        audio_filename: Name of audio file
        result: Prediction result dictionary
        threshold: Confidence threshold for warning
        top_k: Number of top predictions to show (None = all)
    """
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{Style.BRIGHT}PREDICTION RESULT")
    print("=" * 70)

    print(f"\n{Fore.WHITE}{Style.BRIGHT}Audio File: {Fore.YELLOW}{audio_filename}")

    # Main prediction
    predicted_char = result['predicted_character']
    confidence = result['confidence']

    print(f"\n{Fore.WHITE}{Style.BRIGHT}Predicted Character: {Fore.GREEN}{Style.BRIGHT}{predicted_char}")
    print(f"{Fore.WHITE}{Style.BRIGHT}Confidence: {Fore.GREEN}{confidence*100:.2f}%")

    # Confidence warning
    if confidence < threshold:
        print(f"\n{Fore.YELLOW}⚠ Warning: Confidence below threshold ({threshold*100:.0f}%)")
        print(f"{Fore.YELLOW}   The model is uncertain about this prediction.")

    # All probabilities
    print(f"\n{Fore.WHITE}{Style.BRIGHT}All Character Probabilities:")
    print("-" * 70)

    # Sort probabilities
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Apply top_k filter
    if top_k is not None:
        sorted_probs = sorted_probs[:top_k]

    # Find max name length for alignment
    max_name_length = max(len(name) for name, _ in sorted_probs)

    # Print each character with bar
    for rank, (character, prob) in enumerate(sorted_probs, 1):
        # Color based on rank
        if rank == 1:
            color = Fore.GREEN
        elif rank == 2:
            color = Fore.YELLOW
        elif rank == 3:
            color = Fore.CYAN
        else:
            color = Fore.WHITE

        # Create bar
        bar_length = int(prob * 50)  # Scale to 50 characters
        bar = '█' * bar_length

        # Print
        print(f"{color}{rank}. {character:<{max_name_length}} : {prob*100:6.2f}%  {bar}")

    print("=" * 70 + "\n")


def predict_single_file(
    audio_path: str,
    model,
    metadata: dict,
    config,
    device: torch.device,
    threshold: float,
    top_k: int
):
    """Predict single audio file."""
    audio_path = Path(audio_path)

    # Check if file exists
    if not audio_path.exists():
        # Try looking in new_samples directory
        new_samples_path = Path(config.data.new_samples_dir) / audio_path.name
        if new_samples_path.exists():
            audio_path = new_samples_path
        else:
            print(f"\n{Fore.RED}❌ Error: Audio file not found: {audio_path}")
            sys.exit(1)

    print(f"\n{Fore.CYAN}Loading audio file: {audio_path}")

    # Validate audio file
    is_valid, error_msg = validate_audio_file(str(audio_path))
    if not is_valid:
        print(f"\n{Fore.RED}❌ Error: Invalid audio file")
        print(f"{Fore.RED}   {error_msg}")
        sys.exit(1)

    print(f"{Fore.GREEN}✓ Audio file validated")

    # Make prediction
    print(f"{Fore.CYAN}Making prediction...")
    result = predict_audio(model, str(audio_path), metadata, config, device)

    # Print result
    print_prediction_result(audio_path.name, result, threshold, top_k)


def predict_batch(
    model,
    metadata: dict,
    config,
    device: torch.device,
    threshold: float,
    top_k: int
):
    """Predict all files in new_samples directory."""
    new_samples_dir = Path(config.data.new_samples_dir)

    if not new_samples_dir.exists():
        print(f"\n{Fore.RED}❌ Error: new_samples directory not found: {new_samples_dir}")
        sys.exit(1)

    # Find all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(new_samples_dir.glob(f'*{ext}'))
        audio_files.extend(new_samples_dir.glob(f'*{ext.upper()}'))

    if not audio_files:
        print(f"\n{Fore.YELLOW}⚠ No audio files found in {new_samples_dir}")
        sys.exit(0)

    print(f"\n{Fore.CYAN}Found {len(audio_files)} audio files")

    # Predict each file
    for audio_path in sorted(audio_files):
        try:
            # Validate
            is_valid, error_msg = validate_audio_file(str(audio_path))
            if not is_valid:
                print(f"\n{Fore.YELLOW}⚠ Skipping {audio_path.name}: {error_msg}")
                continue

            # Predict
            result = predict_audio(model, str(audio_path), metadata, config, device)

            # Print result
            print_prediction_result(audio_path.name, result, threshold, top_k)

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error processing {audio_path.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Predict character from audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single file
  python scripts/predict.py --audio new_samples/unknown_clip.wav

  # Predict single file (filename only, will look in new_samples/)
  python scripts/predict.py --audio unknown_clip.wav

  # Predict all files in new_samples/
  python scripts/predict.py --batch

  # Show only top 3 predictions
  python scripts/predict.py --audio clip.wav --top_k 3

  # With custom confidence threshold
  python scripts/predict.py --audio clip.wav --threshold 0.7
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to audio file (or filename in new_samples/)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Predict all files in new_samples/ directory'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Confidence threshold (default: from config)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Show only top K predictions (default: all)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.audio and not args.batch:
        print(f"\n{Fore.RED}❌ Error: Must specify either --audio or --batch")
        parser.print_help()
        sys.exit(1)

    if args.audio and args.batch:
        print(f"\n{Fore.RED}❌ Error: Cannot specify both --audio and --batch")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Get threshold
    threshold = args.threshold or config.inference.confidence_threshold

    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{Style.BRIGHT}CHARACTER VOICE CLASSIFICATION - PREDICTION")
    print("=" * 70)

    # Load model
    print(f"\n{Fore.CYAN}Loading model...")
    model, metadata, device = load_model_and_metadata(config)
    print(f"{Fore.GREEN}✓ Model loaded successfully")
    print(f"{Fore.WHITE}  Characters: {', '.join([metadata['idx_to_character'][str(i)] for i in range(metadata['num_characters'])])}")
    print(f"{Fore.WHITE}  Device: {device}")

    # Make predictions
    if args.audio:
        predict_single_file(
            args.audio,
            model,
            metadata,
            config,
            device,
            threshold,
            args.top_k
        )
    else:  # batch mode
        predict_batch(
            model,
            metadata,
            config,
            device,
            threshold,
            args.top_k
        )


if __name__ == "__main__":
    main()
