"""
extract_ticker.py
Run this ONCE from PyCharm/Windows terminal (NOT WSL).
Extracts all rows for a ticker from all CSV files into one file.
Usage: python extract_ticker.py JPM
"""
import pandas as pd
import os
import sys

DATA_DIR = r"Seturi de date NYSE-20260610"
TICKER = sys.argv[1] if len(sys.argv) > 1 else "JPM"
YEARS = range(2001, 2026)

all_rows = []
total_files = 0

for year in YEARS:
    year_dir = os.path.join(DATA_DIR, f"NYSE_{year}")
    if not os.path.isdir(year_dir):
        print(f"  SKIP {year} - no folder")
        continue

    csv_files = sorted([f for f in os.listdir(year_dir) if f.endswith('.csv')])
    for fname in csv_files:
        path = os.path.join(year_dir, fname)
        try:
            df = pd.read_csv(path, usecols=['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            mask = df['Symbol'].str.strip().str.upper() == TICKER.upper()
            rows = df[mask]
            if len(rows) > 0:
                all_rows.append(rows)
            total_files += 1
        except Exception as e:
            pass  # skip malformed files

    print(f"  {year}: {len(csv_files)} files processed")

if not all_rows:
    print(f"No data found for {TICKER}!")
    sys.exit(1)

df = pd.concat(all_rows, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
df = df.dropna(subset=['Date'])
df = df.drop_duplicates(subset=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop(columns=['Symbol'])

out_path = f"data_{TICKER.lower()}_full.csv"
df.to_csv(out_path, index=False)
print(f"\nDone! {len(df)} trading days -> {out_path}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Close range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
