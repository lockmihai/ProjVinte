"""
data_loader.py
Incarca datele unui ticker din fisiere CSV extrase sau arhive ZIP (2001-2026) / Load stock data for a ticker from extracted CSV files or ZIP archives (2001-2026).
Prioritate: foldere extrase > ZIP-uri / Priority: extracted folders > ZIP files.
"""
import pandas as pd
import numpy as np
import zipfile
import os
import io
import gc
from typing import Optional
from config import DATA_DIR, TICKER, YEARS


def _read_ticker_from_csv_bytes(csv_bytes: bytes, ticker: str) -> Optional[pd.DataFrame]:
    """Citeste un CSV dintr-un zip si extrage doar randurile pentru ticker-ul tinta / Read CSV from zip bytes and extract rows matching the target ticker."""
    try:
        # Load the CSV file into memory using specific columns only
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            usecols=['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        # Filter rows corresponding to the target ticker symbol
        mask = df['Symbol'].str.strip().str.upper() == ticker.upper()
        return df[mask].copy()
    except Exception:
        return None


def _read_ticker_from_csv_file(csv_path: str, ticker: str) -> Optional[pd.DataFrame]:
    """Citeste un CSV de pe disc si extrage doar randurile pentru ticker-ul tinta / Read CSV file from disk and filter by target ticker."""
    try:
        df = pd.read_csv(
            csv_path,
            usecols=['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        mask = df['Symbol'].str.strip().str.upper() == ticker.upper()
        return df[mask].copy()
    except Exception:
        return None


def _load_from_extracted(year_dir: str, ticker: str) -> list:
    """Incarca datele dintr-un folder cu CSV-uri extrase / Load all matching CSV files from an extracted year directory."""
    dfs = []
    csv_files = sorted([f for f in os.listdir(year_dir) if f.endswith('.csv')])
    for csv_file in csv_files:
        csv_path = os.path.join(year_dir, csv_file)
        result = _read_ticker_from_csv_file(csv_path, ticker)
        if result is not None and len(result) > 0:
            dfs.append(result)
    return dfs


def _load_from_zip(zip_path: str, ticker: str) -> list:
    """Incarca datele dintr-un fisier ZIP / Load and extract matching data directly from a ZIP file containing multiple CSVs."""
    dfs = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_files = sorted([f for f in zf.namelist() if f.endswith('.csv')])
        for csv_file in csv_files:
            csv_bytes = zf.read(csv_file)
            result = _read_ticker_from_csv_bytes(csv_bytes, ticker)
            if result is not None and len(result) > 0:
                dfs.append(result)
    return dfs


def load_ticker_data(ticker: str = TICKER,
                     years: list = None,
                     zip_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Incarca toate datele istorice pentru un ticker / Load all historical daily stock data for a ticker.

    Incearca mai intai folderele extrase (NYSE_YYYY/), apoi zip-urile / Tries extracted directories first, then ZIP archives.

    Returneaza un DataFrame cu coloanele: Date, Open, High, Low, Close, Volume sortat cronologic / Returns a chronologically sorted DataFrame with OHLCV data.
    """
    if years is None:
        years = YEARS

    all_dfs = []

    for year in years:
        # 1. Try extracted directory path first
        year_dir = os.path.join(zip_dir, f"NYSE_{year}")
        if os.path.isdir(year_dir):
            all_dfs.extend(_load_from_extracted(year_dir, ticker))
            gc.collect() # Run garbage collection to free memory
            continue

        # 2. Fallback to reading from ZIP archive
        zip_name = f"NYSE_{year}.zip"
        zip_path = os.path.join(zip_dir, zip_name)
        if os.path.exists(zip_path):
            all_dfs.extend(_load_from_zip(zip_path, ticker))
            gc.collect()
            continue

        print(f"  !! NYSE_{year}: nu exista nici folder extras, nici ZIP / NYSE_{year}: no folder or ZIP archive exists")

    if not all_dfs:
        raise ValueError(f"Nu s-au gasit date pentru tickerul / No data found for ticker '{ticker}'")

    # Concatenate all daily dataframes into a single DataFrame
    df = pd.concat(all_dfs, ignore_index=True)

    # Parse and clean dates (assuming dd-Mon-YYYY format)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Remove duplicates by date
    df = df.drop_duplicates(subset=['Date'])

    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)

    # Ensure all OHLCV columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # Drop the Symbol column
    df = df.drop(columns=['Symbol'], errors='ignore')

    print(f"  {ticker}: {len(df)} zile ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    print(f"    Close: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    return df


def load_ticker_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Fast path: incarca direct dintr-un CSV pre-extras (contine deja doar ticker-ul) / Fast path: load directly from a pre-filtered JPM CSV.
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    print(f"  {os.path.basename(csv_path)}: {len(df)} zile ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    print(f"    Close: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    return df


def load_jpm_data() -> pd.DataFrame:
    """
    Convenience wrapper: incearca fisierul pre-extras, apoi folderele/ZIP-urile / Convenience wrapper: tries the pre-extracted JPM file first, and falls back to full extraction.
    """
    fast_path = os.path.join(DATA_DIR, '..', 'data_jpm_full.csv')
    if os.path.exists(fast_path):
        return load_ticker_data_from_csv(fast_path)
    return load_ticker_data('JPM')


if __name__ == '__main__':
    # Simple validation test
    df = load_ticker_data('JPM')
    print(df.head(10))
    print(f"\nTotal: {len(df)} rows")
