import os
import pandas as pd
import numpy as np

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ---------------- CONFIG ----------------

EOD_DIR = "EOD"          # your EOD folder
DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

OUT_DIR = "data"         # CSV output folder
MASTER_INDEX_HTML = "docs/index.html"

os.makedirs("docs", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- UTILITIES ----------------

def iter_eod_files(root_dir):
    """Yield (symbol, full_path) for each *_EOD.csv under EOD/."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith("_EOD.csv"):
                symbol = f.replace("_EOD.csv", "")
                full = os.path.join(dirpath, f)
                yield symbol, full


def load_symbol_data(path):
    """Load OHLCV from one EOD file and sort by date."""
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


def compute_atr(df, period=ATR_PERIOD):
    """Standard ATR; kept in case utils_gann uses it."""
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


# ---------------- CORE: DAY1–DAY2–DAY3 LOGIC ----------------

def build_signals_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each swing:
      Day 1: price–time/price–date square (sq_idx)
      Day 2: breakout close vs Day 1 high/low → signal, entry at Close[Day 2]
      Day 3: exit at Close[Day 3]

    Returns a DataFrame with:
      square_date, square_price, direction, square_type,
      signal_date, signal_close,
      exit_date, exit_close,
      position, pnl_points
    """
    results = []
    n = len(df)

    # we need Day3 = sq_idx+2 ⇒ sq_idx <= n-3
    # find_square_* internally scans forward from swing index
    for i in range(n):

        # ---------- FROM SWING LOW → POTENTIAL LONG (UP MOVE SQUARE) ----------
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD,
            )
            if sq_idx is not None and 0 <= sq_idx <= n - 3:
                day1 = sq_idx
                day2 = sq_idx + 1
                day3 = sq_idx + 2

                square_high = df.loc[day1, HIGH_COL]
                square_low = df.loc[day1, LOW_COL]
                square_close = df.loc[day1, CLOSE_COL]
                day2_close = df.loc[day2, CLOSE_COL]
                day3_close = df.loc[day3, CLOSE_COL]

                # Day2 breakout for LONG: Close[Day2] > High[Day1]
                if day2_close > square_high:
                    pnl = day3_close - day2_close  # long, entry at Day2 close, exit at Day3 close
                    results.append({
                        "square_date": df.loc[day1, DATE_COL],
                        "square_price": float(square_close),
                        "square_high": float(square_high),
                        "square_low": float(square_low),
                        "direction": "up",
                        "square_type": sq_type,

                        "signal_date": df.loc[day2, DATE_COL],
                        "signal_close": float(day2_close),

                        "exit_date": df.loc[day3, DATE_COL],
                        "exit_close": float(day3_close),

                        "position": "long",
                        "pnl_points": float(pnl),
                    })

        # ---------- FROM SWING HIGH → POTENTIAL SHORT (DOWN MOVE SQUARE) ----------
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD,
            )
            if sq_idx is not None and 0 <= sq_idx <= n - 3:
                day1 = sq_idx
                day2 = sq_idx + 1
                day3 = sq_idx + 2

                square_high = df.loc[day1, HIGH_COL]
                square_low = df.loc[day1, LOW_COL]
                square_close = df.loc[day1, CLOSE_COL]
                day2_close = df.loc[day2, CLOSE_COL]
                day3_close = df.loc[day3, CLOSE_COL]

                # Day2 breakout for SHORT: Close[Day2] < Low[Day1]
                if day2_close < square_low:
                    pnl = day2_close - day3_close  # short, entry at Day2 close, exit at Day3 close
                    results.append({
                        "square_date": df.loc[day1, DATE_COL],
                        "square_price": float(square_close),
                        "square_high": float(square_high),
                        "square_low": float(square_low),
                        "direction": "down",
                        "square_type": sq_type,

                        "signal_date": df.loc[day2, DATE_COL],
                        "signal_close": float(day2_close),

                        "exit_date": df.loc[day3, DATE_COL],
                        "exit_close": float(day3_close),

                        "position": "short",
                        "pnl_points": float(pnl),
                    })

    if not results:
        cols = [
            "square_date", "square_price", "square_high", "square_low",
            "direction", "square_type",
            "signal_date", "signal_close",
            "exit_date", "exit_close",
            "position", "pnl_points",
        ]
        return pd.DataFrame(columns=cols)

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("square_date").reset_index(drop=True)
    return df_out


# ---------------- HTML RENDERING FOR EACH STOCK ----------------

def render_stock_html(symbol, df_signals: pd.DataFrame) -> str:

    rows = ""
    for _, r in df_signals.iterrows():
        rows += f"""
<tr>
  <td>{r['square_date'].strftime('%Y-%m-%d')}</td>
  <td>{r['square_price']}</td>
  <td>{r['square_high']}</td>
  <td>{r['square_low']}</td>
  <td>{r['direction']}</td>
  <td>{r['square_type']}</td>
  <td>{r['signal_date'].strftime('%Y-%m-%d')}</td>
  <td>{r['signal_close']}</td>
  <td>{r['exit_date'].strftime('%Y-%m-%d')}</td>
  <td>{r['exit_close']}</td>
  <td>{r['position']}</td>
  <td>{r['pnl_points']}</td>
</tr>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} – Gann Square Signals (Day1–Day2–Day3)</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1000px;
  margin: auto;
  padding: 20px;
  background: #fafafa;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
  font-size: 13px;
}}
th, td {{
  padding: 6px 8px;
  border-bottom: 1px solid #ddd;
}}
th {{
  background: #eee;
}}
</style>
</head>
<body>

<h1>{symbol} – Gann Square Signals (Day1 Square, Day2 Signal, Day3 Exit)</h1>

<table>
<tr>
  <th>Square Date (Day1)</th>
  <th>Square Close</th>
  <th>Square High</th>
  <th>Square Low</th>
  <th>Direction</th>
  <th>Square Type</th>
  <th>Signal Date (Day2)</th>
  <th>Signal Close (Entry)</th>
  <th>Exit Date (Day3)</th>
  <th>Exit Close</th>
  <th>Side</th>
  <th>PNL (pts)</th>
</tr>
{rows}
</table>

</body>
</html>
"""
    return html


# ---------------- MASTER INDEX HTML ----------------

def render_master_index(summaries):
    rows = ""
    for s in summaries:
        rows += f"""
<tr>
  <td><a href="{s['link']}">{s['symbol']}</a></td>
  <td>{s['n_signals']}</td>
</tr>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NSE – Gann Square Signals (Day1–Day2–Day3)</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 900px;
  margin: auto;
  padding: 20px;
  background: #fafafa;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
  font-size: 14px;
}}
th, td {{
  padding: 6px 8px;
  border-bottom: 1px solid #ddd;
}}
th {{
  background: #eee;
}}
</style>
</head>
<body>

<h1>NSE Stocks – Gann Square Signals (Day1 Square, Day2 Signal, Day3 Exit)</h1>

<table>
<tr>
  <th>Symbol</th>
  <th># Signals</th>
</tr>
{rows}
</table>

</body>
</html>
"""
    return html


# ---------------- MAIN ----------------

def main():
    summaries = []

    if not os.path.isdir(EOD_DIR):
        print("EOD folder missing.")
        return

    files = list(iter_eod_files(EOD_DIR))
    if not files:
        print("No EOD files found.")
        return

    for symbol, path in sorted(files):
        print(f"Processing {symbol} ...")

        try:
            df = load_symbol_data(path)
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        df = compute_atr(df)
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL)

        sig_df = build_signals_for_symbol(df)

        # CSV output
        out_csv = os.path.join(OUT_DIR, f"{symbol}_signals.csv")
        sig_df.to_csv(out_csv, index=False)

        # HTML per stock
        stock_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(stock_dir, exist_ok=True)
        out_html = os.path.join(stock_dir, "index.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(render_stock_html(symbol, sig_df))

        summaries.append({
            "symbol": symbol,
            "n_signals": len(sig_df),
            "link": f"stocks/{symbol}/index.html",
        })

    # Master index
    master_html = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Done. All Day1–Day2–Day3 signals built.")


if __name__ == "__main__":
    main()
