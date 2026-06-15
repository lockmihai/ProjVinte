"""
config.py
Configurație centralizată pentru proiect / Centralized configuration settings for the project.
"""
import os

# ======== PATHS ========
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory where NYSE dataset CSV files/zips are stored
DATA_DIR = os.path.join(BASE_DIR, "Seturi de date NYSE-20260610")
# Directory where generated plots, metrics, and predicted outputs will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== TICKER PARAMETERS ========
# Stock symbol to predict (e.g., JPM for JPMorgan Chase)
TICKER = "JPM"
# Year range for the dataset (2001 to 2025 inclusive)
YEARS = list(range(2001, 2026))

# ======== PREDICTION CONFIGURATION ========
# Available prediction horizons in days (predicting 1, 5, 10, etc. days ahead)
HORIZONS = [1, 5, 10, 20, 30, 60]  
# Lookback window length (features from the last 60 days are used as input)
SEQ_LENGTH = 60  
# Target column base (predicting Close price behavior)
TARGET_COL = 'Close'
# Prediction task type ('Close' price regression or 'Direction' classification)
TARGET_TYPE = 'Close'  

# ======== FEATURES ========
# Raw price and volume features from the CSVs (non-stationary)
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# List of derived technical indicators that can be used as features
DERIVED_FEATURES = [
    'LogReturn',        # Logarithmic return (stationary)
    'Return',           # Percentage return (stationary)
    'HL_Range',         # Absolute range: High - Low (non-stationary)
    'HL_Range_pct',     # Relative range: (High-Low)/Close (stationary)
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', # Simple Moving Averages (non-stationary)
    'EMA_12', 'EMA_26', # Exponential Moving Averages (non-stationary)
    'RSI_14',           # Relative Strength Index (stationary, bounded 0-100)
    'MACD', 'MACD_Signal', 'MACD_Hist', # Moving Average Convergence Divergence (non-stationary)
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', # Bollinger Bands and relative width (width is stationary)
    'RV_5', 'RV_10', 'RV_20',          # Realized Volatility over 5, 10, 20 days (stationary)
    'Volume_SMA_5', 'Volume_Ratio',     # Volume moving average and relative volume ratio (ratio is stationary)
]

# ======== MODEL HYPERPARAMETERS ========
# LSTM hidden layer dimensionality
HIDDEN_SIZE = 128
# Number of recurrent layers in LSTM
NUM_LAYERS = 2
# Dropout probability applied between LSTM layers
DROPOUT = 0.2
# Maximum training epochs
EPOCHS = 200
# Size of training mini-batches
BATCH_SIZE = 32
# Initial learning rate for optimization
LEARNING_RATE = 1e-3
# Weight decay (L2 regularization) factor
WEIGHT_DECAY = 1e-5
# Early stopping patience: stop if validation loss doesn't improve for N epochs
PATIENCE = 25

# ======== DATA SPLITTING ========
# Percentage of data used for validation
VAL_SPLIT = 0.15
# Percentage of data used for testing
TEST_SPLIT = 0.15
