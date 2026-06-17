"""
evaluate.py
Evaluare model: metrici, grafice, salvare rezultate / Model evaluation: metrics, plots, and saving results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def denormalize(y_scaled, scaler_y):
    """Readuce valorile la scara originală / Scales the values back to their original domain using target scaler."""
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def compute_metrics(y_true, y_pred):
    """
    Calculează metricile de regresie / Computes regression performance metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy (whether the predicted movement direction ↑/↓ matches the actual movement)
    direction_true = np.sign(np.diff(y_true)) if len(y_true) > 1 else np.array([0])
    direction_pred = np.sign(np.diff(y_pred)) if len(y_pred) > 1 else np.array([0])
    directional_acc = np.mean(direction_true == direction_pred) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
        'Directional Accuracy (%)': directional_acc
    }
    return metrics


def plot_loss_history(history, save: bool = True):
    """Graficul loss-ului pe train și validation / Plot train vs validation loss history over training epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=1.5)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Evolutia functiei de pierdere (Loss History)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_history.png'), dpi=150)
    if save:
        plt.close()


def plot_predictions(y_true, y_pred, title: str = 'Predicții vs Valori Reale',
                     save: bool = True):
    """Grafic comparativ între predicții și valorile reale / Plots predicted values against actual values over time."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Valori Reale (Close)', linewidth=1.5, alpha=0.8)
    plt.plot(y_pred, label='Predicții LSTM', linewidth=1.5, alpha=0.8)
    plt.xlabel('Timp (zile) / Time (days)')
    plt.ylabel('Pret Close (USD) / Close Price (USD)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'), dpi=150)
    if save:
        plt.close()


def plot_scatter(y_true, y_pred, save: bool = True):
    """Scatter plot: valori reale vs prezise / Scatter plot comparing actual vs predicted prices."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    # Plot diagonal ideal line (y = x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect Fit')
    plt.xlabel('Valori Reale (USD) / Actual Values (USD)')
    plt.ylabel('Predictii (USD) / Predicted Values (USD)')
    plt.title('Predictii vs Valori Reale (Correlation)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'scatter.png'), dpi=150)
    if save:
        plt.close()


def plot_residuals(y_true, y_pred, save: bool = True):
    """Histograma reziduurilor / Plots a histogram of prediction residuals (errors)."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Eroare (USD) / Error (USD)')
    plt.ylabel('Frecvență / Frequency')
    plt.title('Distributia erorilor de predictie (Residuals)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'residuals.png'), dpi=150)
    if save:
        plt.close()


def evaluate_model(model, X_test, y_test_scaled, scaler_y, history,
                   save_plots: bool = True):
    """
    Pipeline complet de evaluare directă / Complete standard evaluation pipeline.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # De-scale outputs to original space
    y_test_actual = denormalize(y_test_scaled, scaler_y)
    y_pred_actual = denormalize(y_pred_scaled, scaler_y)

    # Compute regression metrics
    metrics = compute_metrics(y_test_actual, y_pred_actual)

    print("\n" + "=" * 50)
    print("REZULTATE EVALUARE / EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    # Generate and save plots
    if save_plots:
        plot_loss_history(history, save=True)
        plot_predictions(y_test_actual, y_pred_actual, save=True)
        plot_scatter(y_test_actual, y_pred_actual, save=True)
        plot_residuals(y_test_actual, y_pred_actual, save=True)

    # Save predictions, actual values, and absolute error difference to CSV
    results_df = np.column_stack([y_test_actual, y_pred_actual, y_test_actual - y_pred_actual])
    header = "Actual_Close,Predicted_Close,Error"
    np.savetxt(os.path.join(OUTPUT_DIR, 'predictions_results.csv'),
               results_df, delimiter=',', header=header, comments='',
               fmt='%.4f')

    return metrics, y_pred_actual


if __name__ == '__main__':
    # Simple validation test
    y_true = np.array([100, 102, 101, 105, 107])
    y_pred = np.array([99, 103, 100, 104, 108])
    m = compute_metrics(y_true, y_pred)
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
