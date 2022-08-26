import torch

# FIXME: Using mps backend causes a slow down in training
device = 'mps' if torch.backends.mps.is_available() else 'cpu'