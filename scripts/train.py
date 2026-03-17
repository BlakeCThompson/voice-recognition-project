"""Training script for character voice classification."""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.data_splitter import load_split
from src.data.dataset import CharacterVoiceDataset, create_dataloader
from src.data.augmentation import get_augmenter
from src.models.wav2vec2_classifier import create_model
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves
import json


def main():
    parser = argparse.ArgumentParser(description="Train character voice classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("CHARACTER VOICE CLASSIFICATION - TRAINING")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.device.use_cuda_if_available else 'cpu')
    print(f"\nDevice: {device}")

    # Load data splits
    print("\nLoading data splits...")
    splits_dir = Path(config.paths.data_splits_dir)

    if not splits_dir.exists():
        print(f"\n❌ Error: Splits directory not found: {splits_dir}")
        print("Please run: python scripts/prepare_data.py")
        sys.exit(1)

    train_dict = load_split(splits_dir / "train.json")
    val_dict = load_split(splits_dir / "val.json")

    print(f"  Train samples: {len(train_dict['data'])}")
    print(f"  Val samples: {len(val_dict['data'])}")
    print(f"  Characters: {train_dict['num_characters']}")

    # Create augmenter (for training only)
    augmenter = get_augmenter(
        config.augmentation._config,
        sample_rate=config.data.sample_rate
    )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CharacterVoiceDataset(
        train_dict,
        sample_rate=config.data.sample_rate,
        duration=config.data.duration,
        target_db=config.preprocessing.target_db,
        silence_threshold_db=config.preprocessing.silence_threshold_db,
        trim_silence=config.preprocessing.trim_silence,
        augmenter=augmenter,
        pad_mode="random"
    )

    val_dataset = CharacterVoiceDataset(
        val_dict,
        sample_rate=config.data.sample_rate,
        duration=config.data.duration,
        target_db=config.preprocessing.target_db,
        silence_threshold_db=config.preprocessing.silence_threshold_db,
        trim_silence=config.preprocessing.trim_silence,
        augmenter=None,  # No augmentation for validation
        pad_mode="center"
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.device.num_workers
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config, train_dict['num_characters'])
    model = model.to(device)

    # Save character mapping
    metadata = {
        'num_characters': train_dict['num_characters'],
        'character_to_idx': train_dict['character_to_idx'],
        'idx_to_character': train_dict['idx_to_character'],
        'model_config': {
            'name': config.model.name,
            'freeze_layers': config.model.freeze_layers,
            'dropout': config.model.dropout,
            'hidden_dims': config.model.classifier_hidden_dims
        }
    }

    metadata_path = Path(config.paths.models_dir) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")

    # Setup optimizer
    print("\nSetting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        [
            {
                'params': model.wav2vec2.parameters(),
                'lr': config.training.learning_rate
            },
            {
                'params': model.classifier.parameters(),
                'lr': config.training.classifier_learning_rate
            }
        ],
        weight_decay=config.training.weight_decay
    )

    # Setup scheduler
    scheduler = None
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
    elif config.training.scheduler == "linear":
        total_steps = len(train_loader) * config.training.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config.training.warmup_steps
        )

    # Setup loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=config.training.gradient_clip,
        accumulation_steps=config.training.accumulation_steps
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train(
        num_epochs=config.training.epochs,
        save_dir=config.paths.models_dir,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every_n_epochs=config.training.save_every_n_epochs
    )

    # Plot training curves
    print("\nGenerating training curves...")
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        trainer.history,
        save_path=results_dir / "training_curves.png"
    )

    print("\n✓ Training complete!")
    print(f"\nBest model saved to: {config.paths.models_dir}/best_model.pt")
    print(f"Results saved to: {config.paths.results_dir}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate model: python scripts/evaluate.py")
    print(f"  2. Make predictions: python scripts/predict.py --audio new_samples/clip.wav")


if __name__ == "__main__":
    main()
