"""Evaluation utilities for model assessment."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple
from tqdm import tqdm
import json


class Evaluator:
    """Evaluator for character voice classification."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        character_names: List[str]
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to run evaluation on
            character_names: List of character names (in order of indices)
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.character_names = character_names

    def evaluate(self) -> Dict:
        """
        Evaluate model on dataset.

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        print("\nEvaluating model...")

        with torch.no_grad():
            for waveforms, labels in tqdm(self.dataloader, desc="Evaluating"):
                waveforms = waveforms.to(self.device)

                # Forward pass
                logits = self.model(waveforms)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Compute metrics
        metrics = self._compute_metrics(
            all_labels,
            all_predictions,
            all_probabilities
        )

        return metrics

    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            labels: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)

        # Precision, recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )

        # Top-k accuracy
        top_3_accuracy = top_k_accuracy_score(labels, probabilities, k=min(3, len(self.character_names)))
        top_5_accuracy = top_k_accuracy_score(labels, probabilities, k=min(5, len(self.character_names)))

        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        # Classification report
        class_report = classification_report(
            labels,
            predictions,
            target_names=self.character_names,
            zero_division=0,
            output_dict=True
        )

        # Package metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'top_3_accuracy': float(top_3_accuracy),
            'top_5_accuracy': float(top_5_accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {},
            'classification_report': class_report
        }

        # Per-class metrics
        for idx, character in enumerate(self.character_names):
            metrics['per_class_metrics'][character] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx])
            }

        return metrics

    def print_metrics(self, metrics: Dict):
        """
        Print evaluation metrics in a formatted way.

        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"  Top-3 Accuracy:     {metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy:     {metrics['top_5_accuracy']:.4f}")

        print(f"\nPer-Character Metrics:")
        print(f"{'Character':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)

        for character, char_metrics in metrics['per_class_metrics'].items():
            print(
                f"{character:<20} "
                f"{char_metrics['precision']:<12.4f} "
                f"{char_metrics['recall']:<12.4f} "
                f"{char_metrics['f1']:<12.4f} "
                f"{char_metrics['support']:<10}"
            )

        print("=" * 60 + "\n")

    def save_metrics(self, metrics: Dict, save_path: str):
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            save_path: Path to save metrics
        """
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {save_path}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    character_names: List[str],
    save_path: str = None
) -> Dict:
    """
    Convenience function to evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run on
        character_names: List of character names
        save_path: Optional path to save metrics

    Returns:
        Metrics dictionary
    """
    evaluator = Evaluator(model, dataloader, device, character_names)
    metrics = evaluator.evaluate()
    evaluator.print_metrics(metrics)

    if save_path is not None:
        evaluator.save_metrics(metrics, save_path)

    return metrics
