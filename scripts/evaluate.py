"""Evaluation script for character voice classification."""

import argparse
import sys
import torch
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.data_splitter import load_split
from src.data.dataset import CharacterVoiceDataset, create_dataloader
from src.models.wav2vec2_classifier import CharacterVoiceClassifier
from src.training.evaluator import evaluate_model
from src.utils.visualization import save_all_visualizations


def main():
    parser = argparse.ArgumentParser(description="Evaluate character voice classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (defaults to best_model.pt)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate on'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("CHARACTER VOICE CLASSIFICATION - EVALUATION")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda_if_available else 'cpu')
    print(f"\nDevice: {device}")

    # Load model path
    model_path = args.model or Path(config.paths.models_dir) / "best_model.pt"

    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found: {model_path}")
        print("Please train a model first: python scripts/train.py")
        sys.exit(1)

    print(f"Model: {model_path}")

    # Load metadata
    metadata_path = Path(config.paths.models_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"\n❌ Error: Metadata not found: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    character_names = [metadata['idx_to_character'][str(i)] for i in range(metadata['num_characters'])]
    print(f"Characters: {', '.join(character_names)}")

    # Load data split
    print(f"\nLoading {args.split} split...")
    split_path = Path(config.paths.data_splits_dir) / f"{args.split}.json"

    if not split_path.exists():
        print(f"\n❌ Error: Split not found: {split_path}")
        print("Please prepare data first: python scripts/prepare_data.py")
        sys.exit(1)

    split_dict = load_split(split_path)
    print(f"  Samples: {len(split_dict['data'])}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = CharacterVoiceDataset(
        split_dict,
        sample_rate=config.data.sample_rate,
        duration=config.data.duration,
        target_db=config.preprocessing.target_db,
        silence_threshold_db=config.preprocessing.silence_threshold_db,
        trim_silence=config.preprocessing.trim_silence,
        augmenter=None,  # No augmentation for evaluation
        pad_mode="center"
    )

    # Create data loader
    dataloader = create_dataloader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers
    )

    # Load model
    print("\nLoading model...")
    model = CharacterVoiceClassifier(
        num_characters=metadata['num_characters'],
        model_name=metadata['model_config']['name'],
        freeze_layers=metadata['model_config']['freeze_layers'],
        dropout=metadata['model_config']['dropout'],
        hidden_dims=metadata['model_config']['hidden_dims']
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation accuracy: {checkpoint.get('best_val_accuracy', 'unknown'):.4f}")

    # Evaluate
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        character_names=character_names,
        save_path=results_dir / f"metrics_{args.split}.json"
    )

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Load training history if evaluating on test set
    if args.split == 'test' and 'history' in checkpoint:
        history = checkpoint['history']
    else:
        history = None

    if history:
        save_all_visualizations(
            history=history,
            metrics=metrics,
            character_names=character_names,
            save_dir=results_dir
        )
    else:
        from src.utils.visualization import (
            plot_confusion_matrix,
            plot_per_character_metrics
        )
        import numpy as np

        # Just save confusion matrix and per-character metrics
        conf_matrix = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(
            conf_matrix,
            character_names,
            save_path=results_dir / f"confusion_matrix_{args.split}.png"
        )
        plot_confusion_matrix(
            conf_matrix,
            character_names,
            save_path=results_dir / f"confusion_matrix_{args.split}_normalized.png",
            normalize=True
        )
        plot_per_character_metrics(
            metrics,
            save_path=results_dir / f"per_character_metrics_{args.split}.png"
        )
        print(f"Visualizations saved to {results_dir}")

    print("\n✓ Evaluation complete!")
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
