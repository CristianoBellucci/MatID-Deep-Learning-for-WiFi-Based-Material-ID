"""Training and evaluation logic for the material identification model."""

import os
import time
import numpy as np
import torch
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

from config import CHECKPOINTS_PATH, PATIENCE, LEN_STACK_PATIENCE, LABELS


class Trainer:
    """Handles model training, validation, and testing."""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = CHECKPOINTS_PATH,
        patience: int = PATIENCE,
        len_stack_patience: int = LEN_STACK_PATIENCE
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer to use
            device: Device to run training on
            checkpoint_dir: Directory to save checkpoints
            patience: Number of epochs to wait before early stopping
            len_stack_patience: Window size for early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.len_stack_patience = len_stack_patience
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=patience//2, 
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Move data to device
            inputs = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                inputs = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.model.compute_loss(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(
        self, 
        preds: List[int], 
        labels: List[int],
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            preds: List of predicted class indices
            labels: List of true class indices
            average: Averaging strategy for metrics
            
        Returns:
            Dictionary of metric names and values
        """
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average=average, zero_division=0),
            'recall': recall_score(labels, preds, average=average, zero_division=0),
            'f1': f1_score(labels, preds, average=average, zero_division=0)
        }
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 100,
        early_stopping: bool = True
    ) -> Tuple[List[float], List[float]]:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs to train for
            early_stopping: Whether to use early stopping
            
        Returns:
            Tuple of (train_losses, val_losses) lists
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}/{num_epochs} - {elapsed:.1f}s - ' \
                  f'train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - ' \
                  f'val_acc: {val_metrics["accuracy"]:.4f} - val_f1: {val_metrics["f1"]:.4f}')
            
            # Check for early stopping
            if early_stopping and self._check_early_stopping(val_loss):
                print(f'Early stopping after {epoch} epochs')
                break
        
        return train_losses, val_losses
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            # Save the best model
            self.save_checkpoint('best_model.pth')
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def save_checkpoint(self, filename: str):
        """Save a checkpoint of the model.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load a checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                inputs = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            **metrics,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation loss over epochs.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def create_trainer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = CHECKPOINTS_PATH,
    patience: int = PATIENCE,
    len_stack_patience: int = LEN_STACK_PATIENCE
) -> Trainer:
    """Create a trainer with default settings.
    
    Args:
        model: The model to train
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        device: Device to run training on
        checkpoint_dir: Directory to save checkpoints
        patience: Patience for early stopping
        len_stack_patience: Window size for early stopping
        
    Returns:
        Initialized Trainer instance
    """
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        len_stack_patience=len_stack_patience
    )
