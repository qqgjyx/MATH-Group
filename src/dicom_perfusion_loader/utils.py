import numpy as np
import torch

def time_to_seconds(t):
    hours, minutes, seconds = int(t[:2]), int(t[2:4]), float(t[4:])
    return 3600 * hours + 60 * minutes + seconds

def intensity_to_concentration(intensity, baseline=None):
    intensity = torch.as_tensor(intensity)  # Convert to PyTorch tensor if it's not already
    if baseline is None:
        baseline = torch.mean(intensity[6:8])
    baseline = baseline + torch.min(baseline)
    # Add a small value to prevent division by zero
    return 1 / (intensity + 1e-10)

# Add more utility functions as needed
