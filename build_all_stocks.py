import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils_swing import detect_swings
from utils_gann import (
    find_square_from_swing_low,
    find_square_from_swing_high
)

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

OUT_DIR = "data"         # outputs CSV here
MASTER_INDEX_HTML = "docs/index.html"

os.makedirs("docs", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- UTILITIES ----------------

def iter_eod_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith("_EOD.csv"):
                symbol = f.replace("_EOD.csv", "")
                full = os.path.join(dirpath, f)
                yield symbol, full


def load_symbol_data(path):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
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


# ---------------- CORE: EXTRACT SQUARES ONLY ----------------

def extract_squares(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    n = len(df)

    for i in range(n):

        # Swing Low → Upward Gann Square
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD
            )
            if sq_idx is not None and 0 <= sq_idx < n:
                results.append({
                    "square_date": df.loc[sq_idx, DATE_COL],
                    "square_price": float(df.loc[sq_idx, CLOSE_COL]),
                    "direction": "up",
                    "square_type": sq_type
                })

        # Swing High → Downward Gann Square
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD
            )
            if sq_idx is not None and 0 <= sq_idx < n:
                results.append({
                    "square_date": df.loc[sq_idx, DATE_COL],
                    "square_price": float(df.loc[sq_idx, CLOSE_COL]),
                    "direction": "down",
                    "square_type": sq_type
                })

    if not results:
        return pd.DataFrame(columns=[
            "square_date", "square_price", "direction", "square_type"
        ])

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("square_date").reset_index(drop=True)
    return out_df


# ---------------- HTML FOR EACH STOCK ----------------

def render_stock_html(symbol, squares_df):

    rows = ""
    for _, r in squares_df.iterrows():
        rows += f"""
<tr>
<td>{r['square_date'].strftime('%Y-%m-%d')}</td>
<td>{r['square_price']}</td>
<td>{r['direction']}</td>
<td>{r['square_type']}</td>
</tr>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} – Gann Square Formations</title>
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
}}

th, td {{
  padding: 8px;
  border-bottom: 1px solid #ccc;
}}

th {{
  background: #eee;
}}
</style>
</head>
<body>

<h1>{symbol} – Gann Square Formations</h1>

<table>
<tr>
<th>Square Date</th>
<th>Square Price</th>
<th>Direction</th>
<th>Square Type</th>
</tr>
{rows}
</table>

</body>
</html>
"""
    return html


# ---------------- HTML MASTER INDEX ----------------

def render_master_index(rows):

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NSE – Gann Square Formations</title>
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
}}

th, td {{
  padding: 8px;
  border-bottom: 1px solid #ccc;
}}

th {{
  background: #eee;
}}
</style>
</head>
<body>

<h1>NSE Stocks – Gann Square Formations</h1>

<table>
<tr>
<th>Symbol</th>
<th># Squares</th>
<th>Link</th>
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
            print(f"Error loading {symbol}: {e}")
            continue

        df = compute_atr(df)
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL)

        squares_df = extract_squares(df)

        # Save CSV
        out_csv = os.path.join(OUT_DIR, f"{symbol}_squares.csv")
        squares_df.to_csv(out_csv, index=False)

        # Save HTML
        stock_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(stock_dir, exist_ok=True)

        out_html = os.path.join(stock_dir, "index.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(render_stock_html(symbol, squares_df))

        summaries.append({
            "symbol": symbol,
            "count": len(squares_df),
            "link": f"stocks/{symbol}/index.html"
        })

    # Build master index
    rows = "\n".join(
        f"<tr><td>{s['symbol']}</td><td>{s['count']}</td><td><a href='{s['link']}'>View</a></td></tr>"
        for s in summaries
    )

    master_html = render_master_index(rows)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Done. All square formations extracted.")


if __name__ == "__main__":
    main()
