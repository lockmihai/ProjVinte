"""
main.py
Punctul de intrare principal al proiectului (v3 - echilibrat) / Main entry point of the project.
NYSE 2001-2025 | JPM (JPMorgan Chase)
Model: LSTM - predictie pret de inchidere (Close) / LSTM model for predicting stock close price.
"""

import matplotlib.pyplot as plt
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
    df = create_targets(df, target_col="Close", horizons=[HORIZON])

    # STATIONARY INPUT FEATURES
    # We use ratios and returns instead of absolute prices to prevent domain-shift
    # where future prices (e.g. 2022-2025) are at completely different scales than historical prices (2001-2018).
    feature_cols = [
        "LogReturn",
        "HL_Range_pct",
        "RSI_14",
        "BB_Width",
        "RV_5",
        "Volume_Ratio",
        "SMA_5_ratio",
        "SMA_20_ratio",
        "MACD_ratio",
        "Open_pct",
        "High_pct",
        "Low_pct",
    ]
    # STATIONARY TARGET
    # Predicting the future return instead of the absolute stock Close price.
    target_cols = [f"target_return_h{HORIZON}"]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target:   Return (t+1)")

    print("\n" + "#" * 50)
    print(f" EXPERIMENT: SEQ_LENGTH = {SEQ_LENGTH} zile, LSTM {NUM_LAYERS} straturi, Hidden size {HIDDEN_SIZE}. learning rate {LEARNING_RATE}, batch size {BATCH_SIZE}, dropout {DROPOUT}")
    print("#" * 50)

    # ============= 3. PREPROCESS =============
    print("\n[3/6] Preprocesare date...")
    data = prepare_data(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_length=SEQ_LENGTH,
        val_split=0.15,
        test_split=0.15,
    )

    # ============= 4. BUILD MODEL =============
    print("\n[4/6] Construire model LSTM...")
    input_size = data["X_train"].shape[2]
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1,
    )
    print(f"  Arhitectura: LSTM({HIDDEN_SIZE}, {NUM_LAYERS} layers)")
    print(f"  Parametri: {count_parameters(model):,}")

    # ============= 5. TRAIN =============
    print("\n[5/6] Antrenare...")
    model, history = train_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        device=device,
    )

    # ============= 6. EVALUATE & RECONSTRUCT PRICES =============
    print("\n[6/6] Evaluare pe setul de test...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(data["X_test"], dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_test_scaled = data["y_test"]
    scaler_y = data["scaler_y"]

    from evaluate import denormalize, compute_metrics

    y_test_return = denormalize(y_test_scaled, scaler_y)
    y_pred_return = denormalize(y_pred_scaled, scaler_y)

    # Reconstruct Close prices
    test_df = data["test_df"]
    close_last_day = test_df["Close"].values[SEQ_LENGTH - 1 : -1]
    y_test_actual_price = close_last_day * (1.0 + y_test_return)
    y_pred_actual_price = close_last_day * (1.0 + y_pred_return)

    # Compute metrics on reconstructed prices
    metrics = compute_metrics(y_test_actual_price, y_pred_actual_price)

    print(f"\nRezultate Evaluare (SEQ={SEQ_LENGTH}):")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    # Generate and save plots
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Evolutia functiei de pierdere (SEQ={SEQ_LENGTH})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_history_seq{SEQ_LENGTH}_{NUM_LAYERS}_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(
        y_test_actual_price, label="Valori Reale (Close)", linewidth=1.5, alpha=0.8
    )
    plt.plot(y_pred_actual_price, label="Predicții LSTM", linewidth=1.5, alpha=0.8)
    plt.xlabel("Timp (zile)")
    plt.ylabel("Pret Close (USD)")
    plt.title(f"Predictii vs Valori Reale (SEQ={SEQ_LENGTH})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"predictions_seq{SEQ_LENGTH}_{NUM_LAYERS}_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_actual_price, y_pred_actual_price, alpha=0.5, s=20)
    min_val = min(y_test_actual_price.min(), y_pred_actual_price.min())
    max_val = max(y_test_actual_price.max(), y_pred_actual_price.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1,
        label="Perfect Fit",
    )
    plt.xlabel("Valori Reale (USD)")
    plt.ylabel("Predictii (USD)")
    plt.title(f"Predictii vs Reale (SEQ={SEQ_LENGTH})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_seq{SEQ_LENGTH}_{NUM_LAYERS}_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(
        y_test_actual_price - y_pred_actual_price,
        bins=30,
        alpha=0.7,
        edgecolor="black",
    )
    plt.xlabel("Eroare (USD)")
    plt.ylabel("Frecvență")
    plt.title(f"Distributia erorilor (SEQ={SEQ_LENGTH})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"residuals_seq{SEQ_LENGTH}_{NUM_LAYERS}_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=150)
    plt.close()

    # Save predictions to CSV with suffix and standard name
    results_df = np.column_stack(
        [
            y_test_actual_price,
            y_pred_actual_price,
            y_test_actual_price - y_pred_actual_price,
        ]
    )
    header = "Actual_Close,Predicted_Close,Error"
    np.savetxt(
        os.path.join(output_dir, f"predictions_results_seq{SEQ_LENGTH}_{NUM_LAYERS}_{HIDDEN_SIZE}_{LEARNING_RATE}_{BATCH_SIZE}_{DROPOUT}.csv"),
        results_df,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.4f",
    )
    np.savetxt(
        os.path.join(output_dir, "predictions_results.csv"),
        results_df,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.4f",
    )

    # ============= SUMMARY =============
    print("\n" + "=" * 60)
    print("  REZUMAT METRICI (LSTM 5 STRATURI, SEQ_LENGTH = 30)")
    print("=" * 60)
    print(
        f"{'SEQ_LENGTH':12s} | {'MSE':10s} | {'RMSE':10s} | {'MAE':10s} | {'MAPE (%)':10s} | {'R2':10s} | {'Dir Acc (%)':12s}"
    )
    print("-" * 88)
    print(
        f"{SEQ_LENGTH:12d} | {metrics['MSE']:10.4f} | {metrics['RMSE']:10.4f} | {metrics['MAE']:10.4f} | {metrics['MAPE (%)']:10.4f} | {metrics['R²']:10.4f} | {metrics['Directional Accuracy (%)']:12.4f}"
    )
    print("=" * 88)

    print("\n" + "=" * 60)
    print("  PROIECT FINALIZAT / PROJECT FINISHED")
    print("  Rezultate salvate in output/ / Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
