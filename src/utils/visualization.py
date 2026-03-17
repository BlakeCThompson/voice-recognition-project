"""Visualization utilities for plotting results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training and validation curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    character_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        character_names: List of character names
        save_path: Optional path to save figure
        normalize: Whether to normalize values
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cbar_label = 'Normalized Count'
    else:
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=character_names,
        yticklabels=character_names,
        cbar_kws={'label': cbar_label},
        ax=ax
    )

    ax.set_xlabel('Predicted Character', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Character', fontsize=12, fontweight='bold')
    ax.set_title('Character Voice Classification Confusion Matrix', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_per_character_metrics(
    metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Plot per-character precision, recall, and F1 scores.

    Args:
        metrics: Metrics dictionary with per_class_metrics
        save_path: Optional path to save figure
    """
    characters = list(metrics['per_class_metrics'].keys())
    precision = [metrics['per_class_metrics'][c]['precision'] for c in characters]
    recall = [metrics['per_class_metrics'][c]['recall'] for c in characters]
    f1 = [metrics['per_class_metrics'][c]['f1'] for c in characters]

    x = np.arange(len(characters))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

    ax.set_xlabel('Character', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Character Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(characters, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-character metrics plot saved to {save_path}")

    plt.close()


def create_prediction_bar(probabilities: Dict[str, float], top_k: int = 5) -> str:
    """
    Create a visual bar chart of prediction probabilities.

    Args:
        probabilities: Dictionary mapping character names to probabilities
        top_k: Number of top predictions to show

    Returns:
        String representation of bar chart
    """
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    output = []
    max_name_length = max(len(name) for name, _ in sorted_probs)

    for rank, (character, prob) in enumerate(sorted_probs, 1):
        bar_length = int(prob * 40)  # Scale to 40 characters max
        bar = '█' * bar_length
        output.append(f"{rank}. {character:<{max_name_length}} : {prob*100:5.2f}%  {bar}")

    return '\n'.join(output)


def save_all_visualizations(
    history: Dict,
    metrics: Dict,
    character_names: List[str],
    save_dir: str
):
    """
    Save all visualizations to directory.

    Args:
        history: Training history
        metrics: Evaluation metrics
        character_names: List of character names
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")

    # Training curves
    plot_training_curves(
        history,
        save_path=save_dir / "training_curves.png"
    )

    # Confusion matrix
    conf_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        conf_matrix,
        character_names,
        save_path=save_dir / "confusion_matrix.png"
    )

    # Normalized confusion matrix
    plot_confusion_matrix(
        conf_matrix,
        character_names,
        save_path=save_dir / "confusion_matrix_normalized.png",
        normalize=True
    )

    # Per-character metrics
    plot_per_character_metrics(
        metrics,
        save_path=save_dir / "per_character_metrics.png"
    )

    print(f"All visualizations saved to {save_dir}")
