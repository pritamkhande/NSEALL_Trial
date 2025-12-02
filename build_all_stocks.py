import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD"                 # Root EOD folder with A,B,...,0-9 subfolders
DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

MASTER_INDEX_HTML = "docs/index.html"
TRADES_CSV_DIR = "data"

os.makedirs("docs", exist_ok=True)
os.makedirs(TRADES_CSV_DIR, exist_ok=True)


# ==========================
# FILE DISCOVERY / DATA LOAD
# ==========================

def iter_eod_files(root_dir: str):
    """
    Yield (symbol, full_path) for all *_EOD.csv under EOD,
    including subfolders like EOD/T/TCS_EOD.csv.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith("_EOD.csv"):
                continue
            symbol = fname.replace("_EOD.csv", "")
            full_path = os.path.join(dirpath, fname)
            yield symbol, full_path


def load_symbol_data(path: str) -> pd.DataFrame:
    """
    Load one symbol CSV:

    Expected columns:
      Symbol, Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df["Date"], errors="coerce")
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


# ==========================
# BACKTEST LOGIC (1-BAR HOLD)
# ==========================

def generate_trades_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each valid Gann square:

      - Signal Date: date at square index (sq_idx)
      - Sq-Type: price_time / price_date
      - Side:
          * swing_high → LONG
          * swing_low  → SHORT
      - Entry Date: Signal Date  (sq_idx)
      - Entry Price: Close[Signal Date]
      - Exit Date: next trading date (sq_idx+1)
      - Exit Price: Close[Exit Date]
      - Profit pts: (Exit - Entry) for long, (Entry - Exit) for short
      - Profit %: Profit pts / EntryPrice * 100
    """
    n = len(df)
    trades: List[Dict] = []

    for i in range(n - 1):  # up to n-2, because we need i+1 as exit
        # LONG from swing_high
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD,
            )
            if sq_idx is not None and sq_idx < n - 1:
                entry_idx = sq_idx
                exit_idx = sq_idx + 1

                entry_date = df.loc[entry_idx, DATE_COL]
                exit_date = df.loc[exit_idx, DATE_COL]
                entry_price = float(df.loc[entry_idx, CLOSE_COL])
                exit_price = float(df.loc[exit_idx, CLOSE_COL])

                profit_pts = exit_price - entry_price
                profit_pct = (profit_pts / entry_price) * 100 if entry_price != 0 else 0.0

                trades.append(
                    {
                        "signal_index": int(sq_idx),
                        "signal_date": entry_date,
                        "square_type": sq_type,
                        "side": "long",
                        "entry_index": int(entry_idx),
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_index": int(exit_idx),
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_pts": profit_pts,
                        "profit_pct": profit_pct,
                    }
                )

        # SHORT from swing_low
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df,
                swing_idx=i,
                date_col=DATE_COL,
                close_col=CLOSE_COL,
                slope_tol=SLOPE_TOL,
                max_lookahead=MAX_LOOKAHEAD,
            )
            if sq_idx is not None and sq_idx < n - 1:
                entry_idx = sq_idx
                exit_idx = sq_idx + 1

                entry_date = df.loc[entry_idx, DATE_COL]
                exit_date = df.loc[exit_idx, DATE_COL]
                entry_price = float(df.loc[entry_idx, CLOSE_COL])
                exit_price = float(df.loc[exit_idx, CLOSE_COL])

                profit_pts = entry_price - exit_price
                profit_pct = (profit_pts / entry_price) * 100 if entry_price != 0 else 0.0

                trades.append(
                    {
                        "signal_index": int(sq_idx),
                        "signal_date": entry_date,
                        "square_type": sq_type,
                        "side": "short",
                        "entry_index": int(entry_idx),
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_index": int(exit_idx),
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_pts": profit_pts,
                        "profit_pct": profit_pct,
                    }
                )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.sort_values(["signal_date", "signal_index"], inplace=True)
        trades_df.reset_index(drop=True, inplace=True)
        trades_df.insert(0, "trade_no", trades_df.index + 1)

    return trades_df


# ==========================
# METRICS (BASED ON 1-BAR TRADES)
# ==========================

def compute_metrics(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
    """
    Metrics:
      - n_trades
      - win_rate (profit_pts > 0)
      - avg_pts
      - avg_pct
      - CAGR (equity compounded on each trade using profit_pct)
      - max_dd (equity drawdown)
      - start_date, end_date, years
    """
    if price_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_pts": 0.0,
            "avg_pct": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "start_date": None,
            "end_date": None,
            "years": 0.0,
        }

    start_date = price_df[DATE_COL].iloc[0]
    end_date = price_df[DATE_COL].iloc[-1]
    years = (end_date - start_date).days / 365.25 if (end_date - start_date).days > 0 else 0.0

    if trades_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_pts": 0.0,
            "avg_pct": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "start_date": start_date,
            "end_date": end_date,
            "years": years,
        }

    n_trades = len(trades_df)
    wins = (trades_df["profit_pts"] > 0).sum()
    win_rate = 100.0 * wins / n_trades
    avg_pts = trades_df["profit_pts"].mean()
    avg_pct = trades_df["profit_pct"].mean()

    # Build simple equity curve assuming full capital compounding
    equity = [1.0]
    for _, row in trades_df.iterrows():
        r = row["profit_pct"] / 100.0
        equity.append(equity[-1] * (1.0 + r))

    equity_arr = np.array(equity)
    if years > 0 and equity_arr[0] > 0:
        cagr = (equity_arr[-1] / equity_arr[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    peaks = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peaks) / peaks
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_pts": avg_pts,
        "avg_pct": avg_pct,
        "cagr": cagr,
        "max_dd": max_dd,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
    }


def build_system_commentary(symbol: str, metrics: Dict, trades_df: pd.DataFrame) -> str:
    n = metrics["n_trades"]
    yrs = metrics["years"]
    win_rate = metrics["win_rate"]
    avg_pts = metrics["avg_pts"]
    avg_pct = metrics["avg_pct"]
    cagr = metrics["cagr"] * 100.0
    max_dd = metrics["max_dd"] * 100.0

    if trades_df.empty:
        return (
            f"For {symbol}, no trades were generated by the Gann square logic. "
            f"This can happen if swing patterns are rare or the slope tolerance is strict."
        )

    if yrs > 0:
        trades_per_year = n / yrs
    else:
        trades_per_year = 0.0

    style = []
    if trades_per_year < 5:
        style.append("very selective")
    elif trades_per_year < 15:
        style.append("moderately active")
    else:
        style.append("frequently trading")

    if max_dd < 5:
        style.append("low drawdown")
    elif max_dd < 12:
        style.append("moderate drawdown")
    else:
        style.append("high drawdown")

    desc = ", ".join(style)

    return (
        f"For {symbol}, the system generated {n} trades over about {yrs:.1f} years, "
        f"with a win rate of {win_rate:.1f}%. The average profit per trade is "
        f"{avg_pts:.2f} points ({avg_pct:.2f}%), leading to an approximate CAGR of "
        f"{cagr:.1f}% with a maximum drawdown of {max_dd:.1f}%. Overall this behaves as a {desc} "
        f"1-bar holding Gann square system (entry on the signal close, exit on next close)."
    )


# ==========================
# HTML RENDERING
# ==========================

def render_stock_html(symbol: str, metrics: Dict, trades_df: pd.DataFrame, commentary: str) -> str:
    start_str = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    years_str = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{symbol} – Gann 1-Bar System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 950px;
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
      table-layout: auto;
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

  <h1>{symbol} – Gann 1-Bar Squaring System</h1>

  <div class="card">
    <h2>Backtest Summary</h2>
    <p>
      <b>Total trades:</b> {metrics["n_trades"]}<br>
      <b>Win rate:</b> {metrics["win_rate"]:.1f}%<br>
      <b>Average profit:</b> {metrics["avg_pts"]:.2f} pts ({metrics["avg_pct"]:.2f}%)<br>
      <b>CAGR (compounded on each trade):</b> {metrics["cagr"]*100:.1f}%<br>
      <b>Max drawdown:</b> {metrics["max_dd"]*100:.1f}%<br>
      <b>Data period:</b> {start_str} to {end_str} ({years_str} yrs)
    </p>
  </div>

  <div class="card">
    <h2>System Commentary</h2>
    <p>{commentary}</p>
  </div>

  <div class="card">
    <h2>All Trades – 1-Bar Holding Logic</h2>
    <p>
      Entry is at the close on the signal date. Exit is at the close on the next trading day.
      Profits are shown in both points and percent of entry price.
    </p>
    <table>
      <tr>
        <th>#</th>
        <th>Signal Date</th>
        <th>Sq-Type</th>
        <th>Side</th>
        <th>Entry Date</th>
        <th>Entry Price</th>
        <th>Exit Date</th>
        <th>Exit Price</th>
        <th>Profit (pts)</th>
        <th>Profit (%)</th>
      </tr>
"""
    for _, row in trades_df.iterrows():
        sig_date = row["signal_date"].strftime("%Y-%m-%d") if pd.notna(row["signal_date"]) else "NA"
        ent_date = row["entry_date"].strftime("%Y-%m-%d")
        ext_date = row["exit_date"].strftime("%Y-%m-%d")

        html += f"""
      <tr>
        <td>{int(row["trade_no"])}</td>
        <td>{sig_date}</td>
        <td>{row["square_type"]}</td>
        <td>{row["side"]}</td>
        <td>{ent_date}</td>
        <td>{row["entry_price"]:.2f}</td>
        <td>{ext_date}</td>
        <td>{row["exit_price"]:.2f}</td>
        <td>{row["profit_pts"]:.2f}</td>
        <td>{row["profit_pct"]:.2f}%</td>
      </tr>
"""

    html += """
    </table>
  </div>

</body>
</html>
"""
    return html


def render_master_index(summaries: List[Dict]) -> str:
    rows_html = ""
    for s in summaries:
        rows_html += f"""
      <tr>
        <td><a href="{s['link']}">{s['symbol']}</a></td>
        <td>{s['n_trades']}</td>
        <td>{s['win_rate']:.1f}%</td>
        <td>{s['avg_pts']:.2f}</td>
        <td>{s['avg_pct']:.2f}%</td>
        <td>{s['cagr']*100:.1f}%</td>
        <td>{s['max_dd']*100:.1f}%</td>
        <td>{s['years']:.1f}</td>
      </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NSE Stocks – Gann 1-Bar System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 950px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
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

  <h1>NSE Stocks – Gann 1-Bar Squaring System</h1>

  <div class="card">
    <p>
      Each symbol below links to a dedicated report. For every detected Gann square,
      the system takes a 1-bar trade: entry on the signal close, exit on the next close.
    </p>
    <table>
      <tr>
        <th>Symbol</th>
        <th># Trades</th>
        <th>Win%</th>
        <th>Avg Pts</th>
        <th>Avg %</th>
        <th>CAGR</th>
        <th>Max DD</th>
        <th>Years</th>
      </tr>
{rows_html}
    </table>
  </div>

</body>
</html>
"""
    return html


# ==========================
# MAIN
# ==========================

def main():
    summaries: List[Dict] = []

    if not os.path.isdir(EOD_DIR):
        print(f"{EOD_DIR} folder not found.")
        return

    files = list(iter_eod_files(EOD_DIR))
    if not files:
        print(f"No *_EOD.csv files found under {EOD_DIR}.")
        return

    for symbol, path in sorted(files, key=lambda x: x[0]):
        print(f"Processing {symbol} from {path} ...")

        try:
            df = load_symbol_data(path)
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            continue

        # Detect swings on OHLC data
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL,
                           lookback_main=1, lookback_fractal=2)

        # Generate 1-bar trades from Gann squares
        trades_df = generate_trades_for_symbol(df)

        # Save trades CSV per symbol
        out_csv = os.path.join(TRADES_CSV_DIR, f"{symbol}_gann_1bar_trades.csv")
        trades_df.to_csv(out_csv, index=False)

        # Compute metrics and commentary
        metrics = compute_metrics(trades_df, df)
        commentary = build_system_commentary(symbol, metrics, trades_df)

        # Per-symbol HTML
        sym_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(sym_dir, exist_ok=True)
        out_html = os.path.join(sym_dir, "index.html")
        html = render_stock_html(symbol, metrics, trades_df, commentary)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        summaries.append(
            {
                "symbol": symbol,
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "avg_pts": metrics["avg_pts"],
                "avg_pct": metrics["avg_pct"],
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "years": metrics["years"],
                "link": f"stocks/{symbol}/index.html",
            }
        )

    # Master index
    master_html = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Master index written to", MASTER_INDEX_HTML)


if __name__ == "__main__":
    main()
