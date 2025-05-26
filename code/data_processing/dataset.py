"""Dataset class for WiFi-based Material Identification."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Union

from config import (
    DEV_PATH, TRAIN_PATH, TEST_PATH, LABELS, LABEL_TO_ID, BATCH_SIZE
)
from .preprocessing import (
    phase_sanitization, amplitude_sanitization,
    prepare_spectrogram_for_resnet
)


class MaterialDataset(Dataset):
    """Dataset class for material identification using WiFi CSI data."""
    
    def __init__(self, mode: str = "train", environment: str = "all"):
        """Initialize the dataset.
        
        Args:
            mode: One of ["train", "dev", "test"]
            environment: One of ["box", "not_box", "all"]
        """
        super().__init__()
        self.mode = mode
        self.environment = environment
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load and preprocess the dataset.
        
        Returns:
            List of data samples with features and labels
        """
        # Determine the base path based on mode
        if self.mode == "train":
            base_path = TRAIN_PATH
        elif self.mode == "dev":
            base_path = DEV_PATH
        else:  # test
            base_path = TEST_PATH
        
        samples = []
        
        # Walk through the directory structure
        for label in os.listdir(base_path):
            if label not in LABELS:
                continue
                
            label_dir = os.path.join(base_path, label)
            if not os.path.isdir(label_dir):
                continue
                
            # Get all .npy files for this label
            for root, _, files in os.walk(label_dir):
                # Check environment filter
                env = "box" if "box" in root else "not_box"
                if self.environment != "all" and env != self.environment:
                    continue
                    
                for file in files:
                    if file.endswith(".npy"):
                        file_path = os.path.join(root, file)
                        samples.append({
                            "file_path": file_path,
                            "label": label,
                            "label_id": LABEL_TO_ID[label],
                            "environment": env
                        })
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        sample = self.data[idx]
        
        # Load CSI data
        csi_data = np.load(sample["file_path"])
        
        # Process the data (simplified - implement based on your actual processing needs)
        # This is where you would apply your signal processing pipeline
        processed_data = self._process_sample(csi_data)
        
        return {
            "data": torch.FloatTensor(processed_data),
            "label": torch.tensor(sample["label_id"], dtype=torch.long),
            "file_path": sample["file_path"]
        }
    
    def _process_sample(self, csi_data: np.ndarray) -> np.ndarray:
        """Process a single CSI sample.
        
        Args:
            csi_data: Raw CSI data
            
        Returns:
            Processed data ready for the model
        """
        # Apply your signal processing pipeline here
        # This is a placeholder - implement based on your needs
        
        # Example processing (simplified):
        # 1. Extract amplitude and phase
        # 2. Apply sanitization
        # 3. Convert to spectrogram
        # 4. Prepare for ResNet input
        
        # For now, just return the data as is (you'll need to implement the actual processing)
        return csi_data
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights
        """
        # Count samples per class
        class_counts = {}
        for sample in self.data:
            label = sample["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate weights (inverse frequency)
        weights = [1.0 / class_counts[LABELS[i]] for i in range(len(LABELS))]
        return torch.FloatTensor(weights)


def create_data_loaders(
    batch_size: int = BATCH_SIZE,
    environment: str = "all",
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for the data loaders
        environment: One of ["box", "not_box", "all"]
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MaterialDataset(mode="train", environment=environment)
    val_dataset = MaterialDataset(mode="dev", environment=environment)
    test_dataset = MaterialDataset(mode="test", environment=environment)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
