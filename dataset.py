"""
dataset.py
PyTorch Dataset wrapper pentru secvențele LSTM / PyTorch Dataset wrapper for LSTM sequences.
"""
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset pentru serii temporale cu secvențe pre-generate.
    Wrapper for pre-generated sliding window sequences of inputs (X) and targets (y).
    """

    def __init__(self, X, y):
        # Convert inputs and targets to PyTorch float tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Return total number of samples (sequences)
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample: (input_sequence, target_value)
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    import numpy as np
    # Simple validation test
    X = np.random.randn(100, 60, 10).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    ds = TimeSeriesDataset(X, y)
    print(f"Dataset size: {len(ds)}, sample shapes: {ds[0][0].shape}, {ds[0][1].shape}")
