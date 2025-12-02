import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# =========================================
# CONFIG
# =========================================

EOD_ROOT = "EOD"        # root with subfolders A, B, 0-9, etc.
DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

MASTER_INDEX_HTML = "docs/index.html"

os.makedirs("docs", exist_ok=True)


# =========================================
# DATA LOADING
# =========================================

def load_symbol_data(path: str) -> pd.DataFrame:
    """
    Load one symbol CSV from EOD/<letter>/<SYMBOL>_EOD.csv

    Expected columns:
      Symbol, Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df["Date"], errors="coerce")
    # Drop timezone if present
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except TypeError:
        pass

    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


# =========================================
# SIGNAL + TRADE LOGIC (T / T+1)
# =========================================

def generate_trades_T_Tplus1(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each swing:

      - From swing LOW:
          use find_square_from_swing_low → SHORT trade
      - From swing HIGH:
          use find_square_from_swing_high → LONG trade

      For each detected square at index sq_idx:
          signal_date = df.Date[sq_idx]
          entry_date  = signal_date
          entry_price = Close on signal_date
          exit_date   = next trading bar (sq_idx + 1)
          exit_price  = Close at sq_idx + 1

      Profit (points):
        LONG  = exit_price - entry_price
        SHORT = entry_price - exit_price
    """
    if df.empty:
        return pd.DataFrame()

    # detect swings first
    df_sw = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL,
                          lookback_main=1, lookback_fractal=2)

    trades = []
    n = len(df_sw)

    for i in range(n):
        # 1) SHORT from swing low (up move square)
        if df_sw.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df_sw, i, DATE_COL, CLOSE_COL,
                slope_tol=0.25,
                max_lookahead=160,
            )
            if sq_idx is not None and sq_idx < n - 1:
                entry_idx = sq_idx
                exit_idx = sq_idx + 1

                entry_date = df_sw.loc[entry_idx, DATE_COL]
                exit_date = df_sw.loc[exit_idx, DATE_COL]
                entry_price = float(df_sw.loc[entry_idx, CLOSE_COL])
                exit_price = float(df_sw.loc[exit_idx, CLOSE_COL])

                profit_pts = entry_price - exit_price  # short

                trades.append(
                    {
                        "trade_no": len(trades) + 1,
                        "side": "short",
                        "square_type": sq_type,
                        "signal_date": entry_date,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_pts": profit_pts,
                    }
                )

        # 2) LONG from swing high (down move square)
        if df_sw.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df_sw, i, DATE_COL, CLOSE_COL,
                slope_tol=0.25,
                max_lookahead=160,
            )
            if sq_idx is not None and sq_idx < n - 1:
                entry_idx = sq_idx
                exit_idx = sq_idx + 1

                entry_date = df_sw.loc[entry_idx, DATE_COL]
                exit_date = df_sw.loc[exit_idx, DATE_COL]
                entry_price = float(df_sw.loc[entry_idx, CLOSE_COL])
                exit_price = float(df_sw.loc[exit_idx, CLOSE_COL])

                profit_pts = exit_price - entry_price  # long

                trades.append(
                    {
                        "trade_no": len(trades) + 1,
                        "side": "long",
                        "square_type": sq_type,
                        "signal_date": entry_date,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_pts": profit_pts,
                    }
                )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("signal_date").reset_index(drop=True)
        trades_df["trade_no"] = np.arange(1, len(trades_df) + 1)

    return trades_df


# =========================================
# METRICS FOR MASTER TABLE
# =========================================

def compute_metrics(trades_df: pd.DataFrame, df_price: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "start_date": df_price[DATE_COL].iloc[0] if not df_price.empty else None,
            "end_date": df_price[DATE_COL].iloc[-1] if not df_price.empty else None,
        }

    n_trades = len(trades_df)
    wins = (trades_df["profit_pts"] > 0).sum()
    win_rate = 100.0 * wins / n_trades
    avg_profit = trades_df["profit_pts"].mean()

    start_date = df_price[DATE_COL].iloc[0]
    end_date = df_price[DATE_COL].iloc[-1]

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "start_date": start_date,
        "end_date": end_date,
    }


# =========================================
# HTML RENDERING
# =========================================

def render_stock_html(symbol: str, metrics: dict, trades_df: pd.DataFrame) -> str:
    start_str = (
        metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] is not None else "N/A"
    )
    end_str = (
        metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] is not None else "N/A"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{symbol} – Gann Squares T/T+1</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Simple Gann square T to T+1 signal log for {symbol}.">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
      line-height: 1.5;
    }}
    h1, h2 {{
      color: #111827;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
  </style>
</head>
<body>

  <h1>{symbol} – Gann Squares T / T+1</h1>
  <div class="card">
    <p>
      Signals generated using swing highs/lows (utils_swing) and Gann square detection
      (utils_gann). Entry is at the closing price on the signal date (T),
      exit is at the closing price of the next trading day (T+1).
    </p>
    <p>
      Data range: {start_str} to {end_str}. Total trades: {metrics["n_trades"]},
      win rate: {metrics["win_rate"]:.1f}%, average profit: {metrics["avg_profit"]:.2f} points.
    </p>
  </div>

  <div class="card">
    <h2>Signals (T / T+1)</h2>
    <table>
      <tr>
        <th>#</th>
        <th>Side</th>
        <th>Square type</th>
        <th>Signal date</th>
        <th>Entry date</th>
        <th>Entry price (Close@T)</th>
        <th>Exit date (T+1)</th>
        <th>Exit price (Close@T+1)</th>
        <th>Profit (points)</th>
      </tr>
"""
    for _, row in trades_df.iterrows():
        html += f"""
      <tr>
        <td>{int(row['trade_no'])}</td>
        <td>{row['side']}</td>
        <td>{row['square_type']}</td>
        <td>{row['signal_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['entry_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['entry_price']:.2f}</td>
        <td>{row['exit_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['exit_price']:.2f}</td>
        <td>{row['profit_pts']:.2f}</td>
      </tr>
"""

    html += """
    </table>
  </div>

</body>
</html>
"""
    return html


def render_master_index(summaries: list[dict]) -> str:
    rows_html = ""
    for s in summaries:
        rows_html += f"""
      <tr>
        <td><a href="{s['link']}">{s['symbol']}</a></td>
        <td>{s['n_trades']}</td>
        <td>{s['win_rate']:.1f}%</td>
        <td>{s['avg_profit']:.2f}</td>
      </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NSE Stocks – Gann T/T+1 Signals</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Simple Gann T to T+1 signal summary for all NSE stocks.">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
    }}
    h1 {{
      color: #111827;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>

  <h1>NSE Stocks – Gann Squares T / T+1</h1>
  <div class="card">
    <p>
      Each stock below has a dedicated page showing all Gann square signals
      with entry on the signal day (Close@T) and exit on the next trading day (Close@T+1).
    </p>
  </div>

  <div class="card">
    <h2>Stock List</h2>
    <table>
      <tr>
        <th>Symbol</th>
        <th># Trades</th>
        <th>Win rate</th>
        <th>Avg profit (points)</th>
      </tr>
{rows_html}
    </table>
  </div>

</body>
</html>
"""
    return html


# =========================================
# MAIN
# =========================================

def main():
    summaries = []

    if not os.path.isdir(EOD_ROOT):
        print(f"EOD root folder '{EOD_ROOT}' not found.")
        return

    # walk subfolders A,B,...,0-9
    for root, _, files in os.walk(EOD_ROOT):
        for fname in files:
            if not fname.endswith("_EOD.csv"):
                continue
            path = os.path.join(root, fname)
            symbol = fname.replace("_EOD.csv", "")
            print(f"Processing {symbol} from {path} ...")

            try:
                df = load_symbol_data(path)
            except Exception as e:
                print(f"  Failed to load {symbol}: {e}")
                continue

            if df.empty:
                print("  No data.")
                continue

            trades_df = generate_trades_T_Tplus1(df)
            metrics = compute_metrics(trades_df, df)

            # per-symbol HTML
            sym_dir = os.path.join("docs", "stocks", symbol)
            os.makedirs(sym_dir, exist_ok=True)
            out_html = os.path.join(sym_dir, "index.html")
            html = render_stock_html(symbol, metrics, trades_df)
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html)

            summaries.append(
                {
                    "symbol": symbol,
                    "n_trades": metrics["n_trades"],
                    "win_rate": metrics["win_rate"],
                    "avg_profit": metrics["avg_profit"],
                    "link": f"stocks/{symbol}/index.html",
                }
            )

    # master index
    master_html = render_master_index(sorted(summaries, key=lambda x: x["symbol"]))
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Master index written to", MASTER_INDEX_HTML)


if __name__ == "__main__":
    main()
