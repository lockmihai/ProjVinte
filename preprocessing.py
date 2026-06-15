"""
preprocessing.py
Feature engineering, creare secvențe, split-uri, normalizare / Feature engineering, sequence creation, splits, normalization.
Suport pentru regresie (preț) și clasificare (direcție) / Support for regression (price) and classification (direction).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă indicatori tehnici complecși / Adds complex technical indicators to the stock DataFrame.
    Expected columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # --- Returns (Stationary) ---
    df["LogReturn"] = np.log(c / c.shift(1)) # Logarithmic daily returns
    df["Return"] = c.pct_change()            # Simple percentage daily returns

    # --- Ranges (Stationary vs Non-stationary) ---
    df["HL_Range"] = h - l                   # Absolute High-Low range (non-stationary)
    df["HL_Range_pct"] = (h - l) / c         # Relative High-Low range (stationary)

    # --- Simple Moving Averages (Non-stationary) ---
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = c.rolling(w).mean()

    # --- Exponential Moving Averages (Non-stationary) ---
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean()

    # --- RSI (14 days, Stationary) ---
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- MACD (Non-stationary indicators) ---
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- Bollinger Bands (20 days) ---
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_Middle"] = sma20
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20 # BB relative width (stationary)

    # --- Realized Volatility over 5, 10, 20 days (Stationary) ---
    for w in [5, 10, 20]:
        df[f"RV_{w}"] = df["Return"].rolling(w).std()

    # --- Volume Indicators (Volume ratio is stationary) ---
    df["Volume_SMA_5"] = v.rolling(5).mean()
    df["Volume_Ratio"] = v / df["Volume_SMA_5"]

    # --- Ratios and deviations (Stationary price features) ---
    # These convert absolute price features into stationary features (percentage distance from moving averages/open/high/low)
    df["SMA_5_ratio"] = c / df["SMA_5"] - 1
    df["SMA_20_ratio"] = c / df["SMA_20"] - 1
    df["MACD_ratio"] = df["MACD"] / c
    df["Open_pct"] = o / c - 1
    df["High_pct"] = h / c - 1
    df["Low_pct"] = l / c - 1

    return df


# ============================================================
# CREARE ȚINTE MULTI-HORIZON / TARGET CREATION
# ============================================================


def create_targets(
    df: pd.DataFrame, target_col: str = "Close", horizons: List[int] = None
) -> pd.DataFrame:
    """
    Creează coloane țintă pentru fiecare orizont de predicție / Creates target columns for each prediction horizon.

    target_h1 = absolute price in t+1
    target_return_h1 = return in t+1 (stationary)
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 30, 60]

    for h in horizons:
        df[f"target_h{h}"] = df[target_col].shift(-h)
        # Calculate stationary return target: (Price(t+h)/Price(t)) - 1
        df[f"target_return_h{h}"] = (df[target_col].shift(-h) / df[target_col]) - 1

    # Direction target (1 day): 1 = price goes up next day, 0 = price goes down/stays same
    df["target_direction"] = (df[target_col].shift(-1) > df[target_col]).astype(int)

    return df


# ============================================================
# SECVENȚE (SLIDING WINDOW) / SEQUENCE GENERATION
# ============================================================


def create_sequences(
    data: np.ndarray, targets: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creează secvențe sliding window / Creates sliding window sequences.

    Args:
        data: (n_samples, n_features) — normalized features
        targets: (n_samples, n_targets) — target values
        seq_length: window length

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences, n_targets)
    """
    X, y = [], []
    # Slide the window across the data
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i, :]) # Extract sequence window
        y.append(targets[i])                  # Extract target for the next step
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================
# PIPELINE PRINCIPAL / MAIN PIPELINE
# ============================================================


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    seq_length: int = 60,
    val_split: float = 0.15,
    test_split: float = 0.15,
    scaler_type: str = "standard",
) -> Dict:
    """
    Pipeline complet de preprocesare / Complete preprocessing pipeline.

    Args:
        df: Input DataFrame
        feature_cols: Features to use as input
        target_cols: Targets to predict
        seq_length: Input sequence window length
        val_split: Validation split fraction
        test_split: Test split fraction
        scaler_type: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    """
    # 1. Drop NaNs caused by rolling calculations and shifts
    df_clean = df.dropna().reset_index(drop=True)
    n = len(df_clean)

    if n < seq_length + 50:
        raise ValueError(
            f"Prea putine date: {n} randuri dupa curatare (minim {seq_length + 50})"
        )

    # 2. Chronological data splitting (no random shuffling to prevent data leakage)
    n_test = max(int(n * test_split), 30)
    n_val = max(int(n * val_split), 30)
    n_train = n - n_val - n_test

    if n_train < seq_length + 30:
        raise ValueError(
            f"Setul de train prea mic: {n_train} (minim {seq_length + 30})"
        )

    train_df = df_clean.iloc[:n_train]
    val_df = df_clean.iloc[n_train : n_train + n_val]
    test_df = df_clean.iloc[n_train + n_val :]

    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")
    print(
        f"  Train: {train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()}"
    )
    print(
        f"  Test:  {test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()}"
    )

    # 3. Extract numpy arrays for features
    train_X_raw = train_df[feature_cols].values.astype(np.float32)
    val_X_raw = val_df[feature_cols].values.astype(np.float32)
    test_X_raw = test_df[feature_cols].values.astype(np.float32)

    # 4. Scale features (Fit only on the training set, transform on all)
    if scaler_type == "minmax":
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

    train_X = scaler_X.fit_transform(train_X_raw)
    val_X = scaler_X.transform(val_X_raw)
    test_X = scaler_X.transform(test_X_raw)

    # 5. Extract and scale targets
    train_y_raw = train_df[target_cols].values.astype(np.float32)
    val_y_raw = val_df[target_cols].values.astype(np.float32)
    test_y_raw = test_df[target_cols].values.astype(np.float32)

    train_y = scaler_y.fit_transform(train_y_raw)
    val_y = scaler_y.transform(val_y_raw)
    test_y = scaler_y.transform(test_y_raw)

    # 6. Create sliding window sequences
    X_train, y_train = create_sequences(train_X, train_y, seq_length)
    X_val, y_val = create_sequences(val_X, val_y, seq_length)
    X_test, y_test = create_sequences(test_X, test_y, seq_length)

    # 7. Keep raw (unscaled) targets for evaluation metrics
    _, y_train_raw_seq = create_sequences(train_X_raw, train_y_raw, seq_length)
    _, y_val_raw_seq = create_sequences(val_X_raw, val_y_raw, seq_length)
    _, y_test_raw_seq = create_sequences(test_X_raw, test_y_raw, seq_length)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Features: {len(feature_cols)}, Targets: {len(target_cols)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_raw": y_train_raw_seq,
        "y_val_raw": y_val_raw_seq,
        "y_test_raw": y_test_raw_seq,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "df_clean": df_clean,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ============================================================
# PREGĂTIRE CLASIFICARE / PREPARE CLASSIFICATION
# ============================================================


def prepare_classification_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 60,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Dict:
    """
    Pregătește datele pentru clasificarea direcției prețului (↑/↓) / Prepares data for binary classification of price direction.
    Binary target: 1 = price goes up next day, 0 = price goes down/stays same.
    """
    df_clean = df.dropna().reset_index(drop=True)
    n = len(df_clean)

    n_test = max(int(n * test_split), 30)
    n_val = max(int(n * val_split), 30)
    n_train = n - n_val - n_test

    train_df = df_clean.iloc[:n_train]
    val_df = df_clean.iloc[n_train : n_train + n_val]
    test_df = df_clean.iloc[n_train + n_val :]

    # Scale features
    scaler_X = StandardScaler()
    train_X = scaler_X.fit_transform(train_df[feature_cols].values.astype(np.float32))
    val_X = scaler_X.transform(val_df[feature_cols].values.astype(np.float32))
    test_X = scaler_X.transform(test_df[feature_cols].values.astype(np.float32))

    # Binary targets are already 0 or 1
    train_y = train_df["target_direction"].values.astype(np.int64)
    val_y = val_df["target_direction"].values.astype(np.int64)
    test_y = test_df["target_direction"].values.astype(np.int64)

    # Create sequences
    X_train, y_train = create_sequences(train_X, train_y.reshape(-1, 1), seq_length)
    X_val, y_val = create_sequences(val_X, val_y.reshape(-1, 1), seq_length)
    X_test, y_test = create_sequences(test_X, test_y.reshape(-1, 1), seq_length)

    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from data_loader import load_ticker_data

    # Simple validation test run
    df = load_ticker_data("JPM")
    df = add_technical_indicators(df)
    df = create_targets(df, target_col="Close")
    print(df.columns.tolist())
    print(df[["Date", "Close", "target_h1", "target_h5", "target_direction"]].tail(10))
