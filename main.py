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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ============= HYPERPARAMETERS =============
    SEQ_LENGTHS = [
        7,
        15,
        30,
    ]  # Test multiple lookback windows (7, 15, and 30 days back)
    HORIZON = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 5  # Deep LSTM (5 layers)
    DROPOUT = 0.25  # Higher dropout to mitigate overfitting in deep recurrent networks
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 25

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

    # Dictionary to store performance metrics for comparison
    results_comparison = {}

    for seq_len in SEQ_LENGTHS:
        print("\n" + "#" * 50)
        print(f" EXPERIMENT: SEQ_LENGTH = {seq_len} zile, LSTM {NUM_LAYERS} straturi")
        print("#" * 50)

        # ============= 3. PREPROCESS =============
        print("\n[3/6] Preprocesare date...")
        data = prepare_data(
            df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            seq_length=seq_len,
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
        close_last_day = test_df["Close"].values[seq_len - 1 : -1]
        y_test_actual_price = close_last_day * (1.0 + y_test_return)
        y_pred_actual_price = close_last_day * (1.0 + y_pred_return)

        # Compute metrics on reconstructed prices
        metrics = compute_metrics(y_test_actual_price, y_pred_actual_price)
        results_comparison[seq_len] = metrics

        print(f"\nRezultate Evaluare (SEQ={seq_len}):")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v:.4f}")

        # Generate and save plots with sequence length suffixes
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
        plt.plot(history["val_loss"], label="Validation Loss", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Evolutia functiei de pierdere (SEQ={seq_len})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"loss_history_seq{seq_len}.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.plot(
            y_test_actual_price, label="Valori Reale (Close)", linewidth=1.5, alpha=0.8
        )
        plt.plot(y_pred_actual_price, label="Predicții LSTM", linewidth=1.5, alpha=0.8)
        plt.xlabel("Timp (zile)")
        plt.ylabel("Pret Close (USD)")
        plt.title(f"Predictii vs Valori Reale (SEQ={seq_len})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"predictions_seq{seq_len}.png"), dpi=150)
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
        plt.title(f"Predictii vs Reale (SEQ={seq_len})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_seq{seq_len}.png"), dpi=150)
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
        plt.title(f"Distributia erorilor (SEQ={seq_len})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"residuals_seq{seq_len}.png"), dpi=150)
        plt.close()

        # Overwrite standard plots with the current run's outputs
        plt.figure(figsize=(14, 6))
        plt.plot(
            y_test_actual_price, label="Valori Reale (Close)", linewidth=1.5, alpha=0.8
        )
        plt.plot(y_pred_actual_price, label="Predicții LSTM", linewidth=1.5, alpha=0.8)
        plt.xlabel("Timp (zile)")
        plt.ylabel("Pret Close (USD)")
        plt.title(f"Predictii vs Valori Reale")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
        plt.plot(history["val_loss"], label="Validation Loss", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Evolutia functiei de pierdere")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
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
        plt.title("Predictii vs Reale")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
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
        plt.title("Distributia erorilor")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals.png"), dpi=150)
        plt.close()

        # Save predictions to CSV with suffix
        results_df = np.column_stack(
            [
                y_test_actual_price,
                y_pred_actual_price,
                y_test_actual_price - y_pred_actual_price,
            ]
        )
        header = "Actual_Close,Predicted_Close,Error"
        np.savetxt(
            os.path.join(output_dir, f"predictions_results_seq{seq_len}.csv"),
            results_df,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.4f",
        )

        # Keep predictions_results.csv as the latest
        np.savetxt(
            os.path.join(output_dir, "predictions_results.csv"),
            results_df,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.4f",
        )

    # ============= COMPARATIVE SUMMARY =============
    print("\n" + "=" * 60)
    print("  REZUMAT COMPARATIV BENCHMARK (LSTM 5 STRATURI)")
    print("=" * 60)
    print(
        f"{'SEQ_LENGTH':12s} | {'MSE':10s} | {'RMSE':10s} | {'MAE':10s} | {'MAPE (%)':10s} | {'R2':10s} | {'Dir Acc (%)':12s}"
    )
    print("-" * 88)
    for seq_len, metrics in results_comparison.items():
        print(
            f"{seq_len:12d} | {metrics['MSE']:10.4f} | {metrics['RMSE']:10.4f} | {metrics['MAE']:10.4f} | {metrics['MAPE (%)']:10.4f} | {metrics['R²']:10.4f} | {metrics['Directional Accuracy (%)']:12.4f}"
        )
    print("=" * 88)

    print("\n" + "=" * 60)
    print("  PROIECT FINALIZAT / PROJECT FINISHED")
    print("  Rezultate salvate in output/ / Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
