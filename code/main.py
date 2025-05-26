"""Main script for WiFi-based Material Identification using Deep Learning."""

import os
import argparse
import json
from datetime import datetime

import torch
import numpy as np

from config import (
    DEVICE, LEARNING_RATE, RESNET_LEARNING_RATE, WEIGHT_DECAY, 
    RESNET_WEIGHT_DECAY, EPOCHS, BATCH_SIZE, LABELS, CHECKPOINTS_PATH
)
from data_processing.dataset import create_data_loaders
from models.network import create_model
from training.trainer import create_trainer
from utils.helpers import (
    set_random_seed, plot_confusion_matrix, save_metrics, plot_learning_curves
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate WiFi-based Material Identification model')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training and evaluation')
    parser.add_argument('--environment', type=str, default='all',
                        choices=['box', 'not_box', 'all'],
                        help='Environment to use for training')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--resnet-lr', type=float, default=RESNET_LEARNING_RATE,
                        help='Learning rate for ResNet layers')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--resnet-weight-decay', type=float, default=RESNET_WEIGHT_DECAY,
                        help='Weight decay for ResNet layers')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default='material_id',
                        help='Name for the experiment')
    parser.add_argument('--save-dir', type=str, default=CHECKPOINTS_PATH,
                        help='Directory to save checkpoints and results')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_experiment(args):
    """Set up the experiment directory and save the configuration."""
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.save_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['device'] = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return exp_dir, config


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Set up experiment directory
    exp_dir, config = setup_experiment(args)
    print(f"Experiment directory: {exp_dir}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        environment=args.environment,
        num_workers=4
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        device=device,
        num_classes=len(LABELS),
        dropout=args.dropout
    )
    
    # Create optimizer with different learning rates for ResNet and other layers
    resnet_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'resnet' in name and 'fc' not in name:  # ResNet layers (except the last FC layer)
            resnet_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.Adam(
        [
            {'params': resnet_params, 'lr': args.resnet_lr, 'weight_decay': args.resnet_weight_decay},
            {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        ]
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=exp_dir,
        patience=args.patience
    )
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping=True
    )
    
    # Plot learning curves
    plot_learning_curves(
        train_losses,
        val_losses,
        save_path=os.path.join(exp_dir, 'learning_curves.png')
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.test(test_loader)
    
    # Save test metrics
    save_metrics(
        test_metrics,
        os.path.join(exp_dir, 'test_metrics.json')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=test_metrics['true_labels'],
        y_pred=test_metrics['pred_labels'],
        classes=LABELS,
        normalize=True,
        save_path=os.path.join(exp_dir, 'confusion_matrix.png')
    )
    
    # Print final metrics
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if metric not in ['confusion_matrix', 'true_labels', 'pred_labels']:
            print(f"{metric}: {value:.4f}")
    
    print(f"\nTraining complete! Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
