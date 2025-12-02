import os
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD"                 # NEW: folder with subfolders A, B, ..., 0-9
EARLY_DIR = "Early_Data"        # optional

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
RISK_PER_TRADE = 0.02
SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

MASTER_INDEX_HTML = "docs/index.html"
TRADES_CSV_DIR = "data"

os.makedirs("docs", exist_ok=True)
os.makedirs(TRADES_CSV_DIR, exist_ok=True)

# ==========================
# UTILITIES
# ==========================

def iter_eod_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith("_EOD.csv"):
                symbol = f.replace("_EOD.csv", "")
                full = os.path.join(dirpath, f)
                yield symbol, full


def load_symbol_data(path):
    df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df["Date"], errors="coerce")
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except Exception:
        pass

    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


def compute_atr(df, period=ATR_PERIOD):
    high = df[HIGH_COL]
    low = df[LOW_COL]
    close = df[CLOSE_COL]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(period, min_periods=1).mean()
    return df


def load_early_close_for_symbol(symbol):
    path = os.path.join(EARLY_DIR, f"{symbol}_early.csv")
    if not os.path.exists(path):
        return None

    edf = pd.read_csv(path)
    edf[DATE_COL] = pd.to_datetime(edf["Date"], errors="coerce")
    try:
        edf[DATE_COL] = edf[DATE_COL].dt.tz_localize(None)
    except:
        pass

    edf = edf.dropna(subset=[DATE_COL, "EarlyClose"])
    edf = edf.sort_values(DATE_COL).reset_index(drop=True)
    return edf


# ==========================
# BACKTEST SUPPORT
# ==========================

def calc_forward_point_profits(df, entry_idx, entry_price, position, max_horizon=4):
    sign = 1 if position == "long" else -1
    out = []
    n = len(df)

    for k in range(max_horizon + 1):
        idx = entry_idx + k
        if idx >= n:
            out.append(np.nan)
        else:
            close_k = df.loc[idx, CLOSE_COL]
            out.append(sign * (close_k - entry_price))
    return out


def calc_tminus1_profit(df, signal_idx, position):
    if signal_idx is None:
        return np.nan
    n = len(df)
    if signal_idx + 1 >= n:
        return np.nan

    sign = 1 if position == "long" else -1
    c0 = df.loc[signal_idx, CLOSE_COL]
    c1 = df.loc[signal_idx + 1, CLOSE_COL]
    return sign * (c1 - c0)


def backtest_symbol(df):
    equity
