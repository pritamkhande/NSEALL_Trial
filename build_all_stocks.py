import os
from datetime import datetime
import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD"                 # folder with A, B, ..., 0-9 subfolders
EARLY_DIR = "Early_Data"        # optional, not used now

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
RISK_PER_TRADE = 0.02           # 2% of equity per trade for equity curve
SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

MASTER_INDEX_HTML = "docs/index.html"
TRADES_CSV_DIR = "data"

os.makedirs("docs", exist_ok=True)
os.makedirs(TRADES_CSV_DIR, exist_ok=True)


# ==========================
# UTILITIES
# ==========================

def iter_eod_files(root_dir: str):
    """
    Yield (symbol, full_path) for all *_EOD.csv under EOD,
    including nested subfolders like EOD/T/TCS_EOD.csv.
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
    Load one symbol CSV.

    Expected columns:
      Symbol, Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df["Date"], errors="coerce")
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except TypeError:
        pass

    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    high = df[HIGH_COL]
    low = df[LOW_COL]
    close = df[CLOSE_COL]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period, min_periods=1).mean()
    return df


# ==========================
# BACKTEST SUPPORT FUNCTIONS
# ==========================

def calc_forward_point_profits(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    position: str,
    max_horizon: int = 4,
) -> list[float]:
    """
    Compute point PnL from entry price to T, T+1,... T+max_horizon closes.
    T here is entry_idx bar.
    """
    sign = 1.0 if position == "long" else -1.0
    pnls = []
    n = len(df)

    for k in range(0, max_horizon + 1):
        idx = entry_idx + k
        if idx >= n:
            pnls.append(np.nan)
        else:
            close_k = df.loc[idx, CLOSE_COL]
            pnl_pts = sign * (close_k - entry_price)
            pnls.append(float(pnl_pts))
    return pnls


def calc_tminus1_profit(
    df: pd.DataFrame,
    signal_idx: int | None,
    position: str,
) -> float:
    """
    Profit from signal-day close to next-day close (T-1 definition used earlier).
    Here we still define T-1 as: close(signal_idx+1) - close(signal_idx)
    for long (reversed sign for short). This keeps the older metric meaning.
    """
    if signal_idx is None:
        return np.nan

    n = len(df)
    if signal_idx + 1 >= n:
        return np.nan

    sign = 1.0 if position == "long" else -1.0
    c0 = df.loc[signal_idx, CLOSE_COL]
    c1 = df.loc[signal_idx + 1, CLOSE_COL]
    pnl_pts = sign * (c1 - c0)
    return float(pnl_pts)


# ==========================
# BACKTEST – NEW ENTRY/EXIT LOGIC
# ==========================
#
# IMPORTANT:
# - Signals (swing + square + breakout condition) are IDENTICAL to earlier.
# - signal_idx and signal_date are the SAME as earlier (square bar index).
# - CHANGE: trade entry is at signal-day CLOSE, and exit is next-day CLOSE.


def backtest_symbol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    trades = []

    # Iterate through all bars that could host a swing
    for i in range(n - 2):  # need room for square + next-day breakout + exit day
        # SHORT setup from swing low
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
                # Breakout condition SAME AS EARLIER:
                # next day close must be below square bar low
                if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:
                    signal_idx = sq_idx
                    signal_date = df.loc[signal_idx, DATE_COL]

                    # NEW RULE:
                    # Entry at signal-day close, exit at next-day close
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1
                    if exit_idx >= n:
                        continue  # safety

                    position = "short"
                    entry_price = df.loc[entry_idx, CLOSE_COL]
                    exit_price = df.loc[exit_idx, CLOSE_COL]

                    # Use same theoretical stop as old system for R calc
                    initial_stop_price = df.loc[sq_idx, HIGH_COL] + 2 * df.loc[sq_idx, "ATR"]
                    stop_price = initial_stop_price

                    # R-multiple
                    risk = initial_stop_price - entry_price
                    pnl = entry_price - exit_price
                    r_mult = pnl / risk if risk != 0 else 0.0

                    pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                    pts_T, pts_T1, pts_T2, pts_T3, pts_T4 = calc_forward_point_profits(
                        df, entry_idx, entry_price, position, max_horizon=4
                    )

                    trades.append(
                        {
                            "trade_no": len(trades) + 1,
                            "signal_index": signal_idx,
                            "signal_date": signal_date,
                            "entry_index": entry_idx,
                            "exit_index": exit_idx,
                            "entry_date": df.loc[entry_idx, DATE_COL],
                            "exit_date": df.loc[exit_idx, DATE_COL],
                            "position": position,
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_price),
                            "initial_stop_price": float(initial_stop_price),
                            "final_stop_price": float(stop_price),
                            "R": float(r_mult),
                            "pnl": float(pnl),
                            "exit_reason": "T+1_close",
                            "square_type": sq_type,
                            "pts_Tm1": pts_Tm1,
                            "pts_T": pts_T,
                            "pts_T1": pts_T1,
                            "pts_T2": pts_T2,
                            "pts_T3": pts_T3,
                            "pts_T4": pts_T4,
                        }
                    )

        # LONG setup from swing high
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
                # Breakout condition SAME AS EARLIER:
                # next day close must be above square bar high
                if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:
                    signal_idx = sq_idx
                    signal_date = df.loc[signal_idx, DATE_COL]

                    # NEW RULE:
                    # Entry at signal-day close, exit at next-day close
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1
                    if exit_idx >= n:
                        continue  # safety

                    position = "long"
                    entry_price = df.loc[entry_idx, CLOSE_COL]
                    exit_price = df.loc[exit_idx, CLOSE_COL]

                    # Same theoretical stop as old system for R calc
                    initial_stop_price = df.loc[sq_idx, LOW_COL] - 2 * df.loc[sq_idx, "ATR"]
                    stop_price = initial_stop_price

                    risk = entry_price - initial_stop_price
                    pnl = exit_price - entry_price
                    r_mult = pnl / risk if risk != 0 else 0.0

                    pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                    pts_T, pts_T1, pts_T2, pts_T3, pts_T4 = calc_forward_point_profits(
                        df, entry_idx, entry_price, position, max_horizon=4
                    )

                    trades.append(
                        {
                            "trade_no": len(trades) + 1,
                            "signal_index": signal_idx,
                            "signal_date": signal_date,
                            "entry_index": entry_idx,
                            "exit_index": exit_idx,
                            "entry_date": df.loc[entry_idx, DATE_COL],
                            "exit_date": df.loc[exit_idx, DATE_COL],
                            "position": position,
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_price),
                            "initial_stop_price": float(initial_stop_price),
                            "final_stop_price": float(stop_price),
                            "R": float(r_mult),
                            "pnl": float(pnl),
                            "exit_reason": "T+1_close",
                            "square_type": sq_type,
                            "pts_Tm1": pts_Tm1,
                            "pts_T": pts_T,
                            "pts_T1": pts_T1,
                            "pts_T2": pts_T2,
                            "pts_T3": pts_T3,
                            "pts_T4": pts_T4,
                        }
                    )

    trades_df = pd.DataFrame(trades)

    # Equity curve (realizing R at exit_date)
    df["equity"] = np.nan
    equity = 1.0
    if not trades_df.empty:
        trades_sorted = trades_df.sort_values("exit_date").reset_index(drop=True)
        t_idx = 0
        n_tr = len(trades_sorted)
    else:
        trades_sorted = None
        t_idx = 0
        n_tr = 0

    for idx in range(n):
        cur_date = df.loc[idx, DATE_COL]
        # Realize all trades whose exit_date <= current date
        while t_idx < n_tr and trades_sorted.loc[t_idx, "exit_date"] <= cur_date:
            r_mult = trades_sorted.loc[t_idx, "R"]
            risk_amount = equity * RISK_PER_TRADE
            equity += r_mult * risk_amount
            t_idx += 1
        df.loc[idx, "equity"] = equity

    return trades_df, df


# ==========================
# METRICS + COMMENTARY
# ==========================

def compute_metrics(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "start_date": None,
            "end_date": None,
            "years": 0.0,
        }

    n_trades = len(trades_df)
    wins = (trades_df["R"] > 0).sum()
    win_rate = 100.0 * wins / n_trades
    avg_R = trades_df["R"].mean()

    eq = price_df["equity"].dropna()
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]
    start_date = price_df[DATE_COL].iloc[0]
    end_date = price_df[DATE_COL].iloc[-1]
    years = (end_date - start_date).days / 365.25
    if years > 0 and start_eq > 0:
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    equity_arr = eq.values
    peaks = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peaks) / peaks
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "cagr": cagr,
        "max_dd": max_dd,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
    }


def build_system_commentary(symbol: str, metrics: dict, trades_df: pd.DataFrame) -> str:
    n = metrics["n_trades"]
    years = metrics["years"] or 0.0
    avg_R = metrics["avg_R"]
    win_rate = metrics["win_rate"]
    cagr = metrics["cagr"] * 100
    max_dd = metrics["max_dd"] * 100

    if years > 0:
        trades_per_year = n / years
    else:
        trades_per_year = 0.0

    if trades_df.empty:
        return f"No trades were generated for {symbol} under this T+1 close rule."

    avg_hold = (trades_df["exit_index"] - trades_df["entry_index"]).mean()

    style = []
    if trades_per_year < 5:
        style.append("very selective system")
    elif trades_per_year < 15:
        style.append("moderately active swing system")
    else:
        style.append("active swing/position system")

    if max_dd < 5:
        style.append("with very conservative risk")
    elif max_dd < 12:
        style.append("with moderate risk")
    else:
        style.append("with aggressive risk")

    if cagr < 2:
        style.append("designed more for research than raw returns")
    elif cagr < 8:
        style.append("balanced between robustness and return")
    else:
        style.append("tilted towards maximising return")

    style_txt = ", ".join(style)

    return (
        f"For {symbol}, the system generated {n} trades over the full sample "
        f"({years:.1f} years), averaging about {trades_per_year:.1f} trades per year. "
        f"The holding period is fixed at 1 bar (entry at signal close, exit at next-day close). "
        f"With a win rate of {win_rate:.1f}% and an average outcome of {avg_R:.2f}R per trade, "
        f"the equity curve grows at roughly {cagr:.1f}% CAGR while suffering a maximum drawdown "
        f"of {max_dd:.1f}%. Overall, this behaves like a {style_txt}."
    )


# ==========================
# HTML RENDERING
# ==========================

def render_stock_html(symbol: str, metrics: dict, trades_df: pd.DataFrame, commentary: str) -> str:
    start_str = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    years_str = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{symbol} – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Mechanical Gann Price-Time and Price-Date Squaring backtest on {symbol} daily data. Entry at signal-day close, exit at next-day close.">
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
    h1, h2, h3 {{
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

  <h1>{symbol} – Gann Squaring System</h1>
  <p>
    Fully mechanical backtest of a Price-Time / Price-Date Squaring system inspired by W.D. Gann,
    applied to {symbol} daily data from {start_str} to {end_str}.
    Trades enter at the signal-day closing price and exit at the next day closing price.
  </p>

  <div class="card">
    <h2>Backtest Summary</h2>
    <p>
      <b>Number of trades:</b> {metrics["n_trades"]}<br>
      <b>Win rate:</b> {metrics["win_rate"]:.1f}%<br>
      <b>Average R per trade:</b> {metrics["avg_R"]:.2f} R<br>
      <b>CAGR (normalized equity):</b> {metrics["cagr"]*100:.1f}%<br>
      <b>Maximum drawdown:</b> {metrics["max_dd"]*100:.1f}%<br>
      <b>Test length:</b> {years_str} years
    </p>
  </div>

  <div class="card">
    <h2>System Behaviour Commentary</h2>
    <p>{commentary}</p>
  </div>

  <div class="card">
    <h2>All Trades (T+1 close exits)</h2>
    <table>
      <tr>
        <th>#</th>
        <th>Signal date</th>
        <th>Entry date</th>
        <th>Entry price</th>
        <th>Exit date</th>
        <th>Exit price</th>
        <th>Side</th>
        <th>R</th>
        <th>Square type</th>
        <th>Exit reason</th>
        <th>T(-1)</th>
        <th>T</th>
        <th>T+1</th>
        <th>T+2</th>
        <th>T+3</th>
        <th>T+4</th>
      </tr>
"""
    for _, row in trades_df.iterrows():
        trade_no = int(row["trade_no"])
        sig_date = row["signal_date"].strftime('%Y-%m-%d') if pd.notna(row["signal_date"]) else "NA"

        html += f"""
      <tr>
        <td>{trade_no}</td>
        <td>{sig_date}</td>
        <td>{row['entry_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['entry_price']:.2f}</td>
        <td>{row['exit_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['exit_price']:.2f}</td>
        <td>{row['position']}</td>
        <td>{row['R']:.2f}</td>
        <td>{row['square_type']}</td>
        <td>{row['exit_reason']}</td>
        <td>{row['pts_Tm1']:.2f}</td>
        <td>{row['pts_T']:.2f}</td>
        <td>{row['pts_T1']:.2f}</td>
        <td>{row['pts_T2']:.2f}</td>
        <td>{row['pts_T3']:.2f}</td>
        <td>{row['pts_T4']:.2f}</td>
      </tr>
"""

    html += """
    </table>
  </div>

  <div class="footer" style="font-size:12px;color:#6b7280;margin-top:24px;">
    This is a research backtest ignoring trading costs, slippage and execution constraints.
    It is not trading advice.
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
        <td>{s['avg_R']:.2f}</td>
        <td>{s['cagr']*100:.1f}%</td>
        <td>{s['max_dd']*100:.1f}%</td>
        <td>{s['years']:.1f}</td>
      </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NSE Stocks – Gann T+1 Close System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Gann Price-Time / Price-Date Squaring backtest for all NSE stocks with entry at signal close and exit at next-day close.">
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

  <h1>NSE Stocks – Gann T+1 Close System</h1>
  <div class="card">
    <p>
      Each stock below has a dedicated backtest report for the Gann Price-Time / Price-Date Squaring system,
      with trades entering at the signal-day closing price and exiting at the next-day closing price.
    </p>
  </div>

  <div class="card">
    <h2>Stock List</h2>
    <table>
      <tr>
        <th>Symbol</th>
        <th># Trades</th>
        <th>Win rate</th>
        <th>Avg R</th>
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
    summaries = []

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

        df = compute_atr(df)
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL,
                           lookback_main=1, lookback_fractal=2)

        trades_df, price_df = backtest_symbol(df)

        # Save trades CSV per symbol
        out_csv = os.path.join(TRADES_CSV_DIR, f"{symbol}_gann_trades_Tplus1.csv")
        trades_df.to_csv(out_csv, index=False)

        metrics = compute_metrics(trades_df, price_df)
        commentary = build_system_commentary(symbol, metrics, trades_df)

        # per-symbol HTML
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
                "avg_R": metrics["avg_R"],
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "years": metrics["years"],
                "link": f"stocks/{symbol}/index.html",
            }
        )

    master_html = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Master index written to", MASTER_INDEX_HTML)


if __name__ == "__main__":
    main()
