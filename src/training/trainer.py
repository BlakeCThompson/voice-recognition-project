"""Training loop and utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import json


class Trainer:
    """Trainer class for character voice classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            gradient_clip: Gradient clipping max norm
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps

        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (waveforms, labels) in enumerate(progress_bar):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(waveforms)
            loss = self.criterion(logits, labels)

            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Accumulate loss (multiply back by accumulation_steps)
            total_loss += loss.item() * self.accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{correct / total:.4f}'
            })

        # Learning rate scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Calculate average metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]  ")

        with torch.no_grad():
            for waveforms, labels in progress_bar:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(waveforms)
                loss = self.criterion(logits, labels)

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
                    'acc': f'{correct / total:.4f}'
                })

        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 5,
        save_every_n_epochs: int = 5
    ):
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            save_every_n_epochs: Save checkpoint every N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("TRAINING START")
        print("=" * 60)

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}  |  Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}  |  Val Acc:   {val_metrics['accuracy']:.4f}")

            # Save best model
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.save_checkpoint(save_dir / "best_model.pt", is_best=True)
                print(f"  ✓ New best model saved (Val Acc: {self.best_val_accuracy:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = save_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(checkpoint_path)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best validation accuracy: {self.best_val_accuracy:.4f}")
                break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")

        # Save final history
        self.save_history(save_dir / "training_history.json")

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def save_history(self, path: Path):
        """
        Save training history to JSON.

        Args:
            path: Path to save history
        """
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path: Path):
        """
        Load checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
