"""Configuration file for the WiFi-based Material Identification project."""

import os

# Set the seed for reproducibility
SEED = 104

# Path configurations
ROOT = "DATASET ROOT PATH"
DATASETS_PATH = os.path.join(ROOT, "Dataset")
DEV_PATH = os.path.join(DATASETS_PATH, "dev")
TRAIN_PATH = os.path.join(DATASETS_PATH, "train")
TEST_PATH = os.path.join(DATASETS_PATH, "test")
CHECKPOINTS_PATH = os.path.join(ROOT, "ckp")

# Model hyperparameters
LEARNING_RATE = 0.0001
RESNET_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
RESNET_WEIGHT_DECAY = 0.0001
EPOCHS = 100
DROPOUT = 0.2
BATCH_SIZE = 64
EMBEDDING_DIM = 128

# Training parameters
PATIENCE = 5
LEN_STACK_PATIENCE = 4

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class labels and mappings
LABELS = [
    "acrylic", "aluminium", "brass", "copper", "empty", "lignum_vitae",
    "nylon", "oak", "pine", "pp", "pvc", "rose_wood", "steel"
]

ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

# Signal processing parameters
DEFAULT_WINDOW_SIZE = 91
DEFAULT_SAMPLINGS = 1068
DEFAULT_RESAMPLING_RATIO = 15
EPSILON = 3  # For amplitude sanitization
