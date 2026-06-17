"""
main.py
Punctul de intrare principal al proiectului (v3 - echilibrat) / Main entry point of the project.
NYSE 2001-2025 | JPM (JPMorgan Chase)
Model: LSTM - predictie pret de inchidere (Close) / LSTM model for predicting stock close price.
"""
import torch
import numpy as np
import os
import sys

# Insert current file's directory path to sys.path to enable local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_jpm_data
from preprocessing import add_technical_indicators, create_targets, prepare_data
from model import LSTMPredictor, count_parameters
from train import train_model
from evaluate import evaluate_model


def main():
    print("=" * 60)
    print("  PREDICTIE PRET ACTIUNI CU LSTM")
    print("  NYSE 2001-2025 | JPMorgan Chase (JPM)")
    print("=" * 60)

    # Use GPU (CUDA) if available, otherwise fallback to CPU
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nDevice: {device}")

    # ============= HYPERPARAMETERS =============
    SEQ_LENGTH = 30       # Lookback window: number of past days used as input features
    HORIZON = 1           # Prediction horizon: predict 1 day into the future
    HIDDEN_SIZE = 16      # LSTM hidden state vector size
    NUM_LAYERS = 4        # Deep LSTM (5 layers)
    DROPOUT = 0.2         # Higher dropout to mitigate overfitting in deep recurrent networks
    EPOCHS = 150          # Maximum epochs to train the model
    BATCH_SIZE = 32       # Batch size for SGD optimizer
    LEARNING_RATE = 1e-3  # Step size for optimizer
    WEIGHT_DECAY = 1e-5   # L2 regularization strength
    PATIENCE = 25         # Early stopping patience in epochs
    # ============= 1. LOAD DATA =============
    print("\n[1/6] Incarcare date...")
    # Load JPM stock historical price data (loads from local data_jpm_full.csv fast-path)
    df = load_jpm_data()

    # ============= 2. FEATURE ENGINEERING =============
    print("\n[2/6] Feature engineering + tinte...")
    # Compute derived indicators (RSI, Bollinger Bands, Moving Averages, Volatility, Volume Ratio)
    df = add_technical_indicators(df)
    # Generate shifting targets for regression and classification tasks
    df = create_targets(df, target_col='Close', horizons=[HORIZON])

    # STATIONARY INPUT FEATURES
    # We use ratios and returns instead of absolute prices to prevent domain-shift 
    # where future prices (e.g. 2022-2025) are at completely different scales than historical prices (2001-2018).
    feature_cols = [
        'LogReturn', 'HL_Range_pct', 'RSI_14', 'BB_Width',
        'RV_5', 'Volume_Ratio', 'SMA_5_ratio', 'SMA_20_ratio',
        'MACD_ratio', 'Open_pct', 'High_pct', 'Low_pct',
    ]
    # STATIONARY TARGET
    # Predicting the future return instead of the absolute stock Close price.
    target_cols = [f'target_return_h{HORIZON}']

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target:   Return (t+1)")

    # ============= 3. PREPROCESS =============
    print("\n[3/6] Preprocesare date...")
    # Normalize features using StandardScaler, perform chronological train/val/test splits, 
    # and create sliding window input/output sequences.
    data = prepare_data(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_length=SEQ_LENGTH,
        val_split=0.15,
        test_split=0.15
    )

    # ============= 4. BUILD MODEL =============
    print("\n[4/6] Construire model LSTM...")
    input_size = data['X_train'].shape[2]
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1
    )
    print(f"  Arhitectura: LSTM({HIDDEN_SIZE}, {NUM_LAYERS} layer)")
    print(f"  Parametri: {count_parameters(model):,}")

    # ============= 5. TRAIN =============
    print("\n[5/6] Antrenare...")
    # Train the model with early stopping based on validation loss to prevent overfitting.
    model, history = train_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        device=device
    )

    # ============= 6. EVALUATE & RECONSTRUCT PRICES =============
    print("\n[6/6] Evaluare pe setul de test...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # De-scale returns to their raw percentage domain
    y_test_scaled = data['y_test']
    scaler_y = data['scaler_y']
    
    from evaluate import denormalize, compute_metrics, plot_loss_history, plot_predictions, plot_scatter, plot_residuals
    y_test_return = denormalize(y_test_scaled, scaler_y)
    y_pred_return = denormalize(y_pred_scaled, scaler_y)

    # Get Close price of the last day in each test window
    # Because target return target_return_h1 represents (Close(t+1)/Close(t)) - 1,
    # the base price Close(t) is the actual Close price of the final day in our lookback sequence window.
    test_df = data['test_df']
    close_last_day = test_df['Close'].values[SEQ_LENGTH - 1 : -1]

    # Reconstruct the absolute actual and predicted Close prices for evaluation metrics
    y_test_actual_price = close_last_day * (1.0 + y_test_return)
    y_pred_actual_price = close_last_day * (1.0 + y_pred_return)

    # Compute standard regression metrics (MSE, MAE, MAPE, R2, Trend Directional Accuracy) on reconstructed prices
    metrics = compute_metrics(y_test_actual_price, y_pred_actual_price)

    print("\n" + "=" * 50)
    print("REZULTATE EVALUARE (PE PRETURI RECONSTRUITE) / EVALUATION ON RECONSTRUCTED PRICES")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    # Generate and save evaluation plots into the output directory
    plot_loss_history(history, save=True)
    plot_predictions(y_test_actual_price, y_pred_actual_price, save=True)
    plot_scatter(y_test_actual_price, y_pred_actual_price, save=True)
    plot_residuals(y_test_actual_price, y_pred_actual_price, save=True)

    # Save absolute prediction results, targets, and errors to CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    results_df = np.column_stack([y_test_actual_price, y_pred_actual_price, y_test_actual_price - y_pred_actual_price])
    header = "Actual_Close,Predicted_Close,Error"
    np.savetxt(os.path.join(output_dir, 'predictions_results.csv'),
               results_df, delimiter=',', header=header, comments='',
               fmt='%.4f')

    print("\n" + "=" * 60)
    print("  PROIECT FINALIZAT / PROJECT FINISHED")
    print("  Rezultate salvate in output/ / Results saved in output/")
    print("=" * 60)


if __name__ == '__main__':
    main()
