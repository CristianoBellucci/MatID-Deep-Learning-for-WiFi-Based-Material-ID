"""Utility functions for the WiFi-based Material Identification project."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    classes: List[str],
    normalize: bool = False,
    title: str = 'Confusion matrix',
    cmap: Any = plt.cm.Blues,
    save_path: Optional[str] = None
) -> None:
    """Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        cmap: Colormap for the plot
        save_path: Path to save the plot (optional)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes
    )
    
    # Set labels and title
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def save_metrics(
    metrics: Dict[str, Any], 
    filepath: str
) -> None:
    """Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        filepath: Path to save the metrics file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary containing the loaded metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def set_random_seed(seed: int = 104) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation learning curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def print_metrics(metrics: Dict[str, float], prefix: str = '') -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics to print
        prefix: Prefix to add to each metric name
    """
    for name, value in metrics.items():
        if name != 'confusion_matrix':
            print(f"{prefix}{name}: {value:.4f}")


def get_class_distribution(dataset) -> Dict[str, int]:
    """Get the distribution of classes in a dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary mapping class names to counts
    """
    from collections import defaultdict
    
    class_counts = defaultdict(int)
    for sample in dataset:
        label = sample['label']
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] += 1
    
    return dict(class_counts)
