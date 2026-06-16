"""
train.py
Training loop cu early stopping, scheduler, logging / Training loop with early stopping, scheduler, logging.
Suport pentru regresie și clasificare / Support for regression and classification.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import TimeSeriesDataset
from typing import Dict, Tuple, Optional


def train_model(model: nn.Module,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 200,
                batch_size: int = 32,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-5,
                patience: int = 25,
                device: str = 'cpu',
                task: str = 'regression',
                loss_type: str = 'mse') -> Tuple[nn.Module, Dict]:
    """
    Antrenare model cu early stopping / Train neural network model with early stopping.

    Args:
        task: 'regression' sau / or 'classification'
    """
    model.to(device)

    # 1. Create PyTorch datasets and DataLoader objects for batching
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Select Loss Criterion based on task type and option
    if task == 'classification':
        criterion = nn.BCELoss() # Binary Cross Entropy Loss for classification
    elif loss_type == 'huber':
        criterion = nn.HuberLoss(delta=1.0) # Huber loss (robust to outliers)
    else:
        criterion = nn.MSELoss() # Mean Squared Error loss for regression

    # 3. Setup optimizer and dynamic learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # 4. Initialize training state variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- TRAIN STEP ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad() # Clear gradients from previous step
            y_pred = model(X_batch)

            if task == 'classification':
                y_pred = y_pred.squeeze()
                y_batch = y_batch.float().squeeze()

            loss = criterion(y_pred, y_batch)
            loss.backward() # Backpropagation to compute gradients
            
            # Gradient clipping to prevent exploding gradients in recurrent network
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # Update model weights

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- VALIDATION STEP ---
        model.eval()
        val_losses = []
        with torch.no_grad(): # Disable gradient computation to save memory and compute
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)

                if task == 'classification':
                    y_pred = y_pred.squeeze()
                    y_batch = y_batch.float().squeeze()

                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Step scheduler based on validation loss progression
        scheduler.step(avg_val_loss)

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Clone and save the state dict of the best model iteration
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Periodically log training and validation metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Pat: {patience_counter}/{patience}")

        # Stop training early if validation loss hasn't improved for 'patience' epochs
        if patience_counter >= patience:
            print(f"  Early stopping @ epoch {epoch+1}")
            break

    # Load weights of the best performing model epoch
    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, history
