import torch

# -------------------------------------------------------
# Device configuration
# -------------------------------------------------------
# Whether CUDA is available; used throughout the training scripts
use_cuda = torch.cuda.is_available()
# Default device: "cuda" if available, else fall back to "cpu"
device = torch.device("cuda" if use_cuda else "cpu")

# -------------------------------------------------------
# Data loader settings
# -------------------------------------------------------
# Number of worker threads for each DataLoader
n_threads = 4
# Batch size for training (number of samples per iteration)
batch_size = 2

# -------------------------------------------------------
# Randomness control
# -------------------------------------------------------
# Global random seed for reproducibility
random_seed = 1337

# -------------------------------------------------------
# Optimization settings
# -------------------------------------------------------
# Initial learning rate for self-supervised / task-specific training
lr_self = 1e-4
# Minimum learning rate (for schedulers that decay LR)
min_lr = 1e-6
# Weight decay for Adam optimizer (L2 regularization strength)
weight_decay = 2e-5

# -------------------------------------------------------
# Training schedule
# -------------------------------------------------------
# Total number of training epochs
epochs = 100
# Epoch index from which model saving/checkpointing begins
save_epoch = 0

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
# Base directory for saving logs, checkpoints, and results
data_dir = './'