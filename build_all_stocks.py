import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD"
EARLY_DIR = "Early_Data"

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
    if signal_idx + 1 >= len(df):
        return np.nan

    sign = 1 if position == "long" else -1
    c0 = df.loc[signal_idx, CLOSE_COL]
    c1 = df.loc[signal_idx + 1, CLOSE_COL]
    return sign * (c1 - c0)
def backtest_symbol(df: pd.DataFrame):

    trades = []
    n = len(df)

    for i in range(n - 2):

        # ============ SHORT ==============
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df, i, DATE_COL, CLOSE_COL,
                slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )

            if sq_idx is not None and sq_idx < n - 2:

                if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:

                    signal_idx = sq_idx + 1
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1

                    entry_price = float(df.loc[entry_idx, CLOSE_COL])
                    exit_price = float(df.loc[exit_idx, CLOSE_COL])

                    initial_stop = df.loc[sq_idx, HIGH_COL] + 2 * df.loc[sq_idx, "ATR"]
                    risk = initial_stop - entry_price
                    pnl = entry_price - exit_price
                    R = pnl / risk if risk != 0 else 0.0

                    pts_Tm1 = calc_tminus1_profit(df, signal_idx, "short")
                    pts = calc_forward_point_profits(df, entry_idx, entry_price, "short")

                    trades.append({
                        "trade_no": len(trades) + 1,
                        "square_date": df.loc[sq_idx, DATE_COL],
                        "signal_date": df.loc[signal_idx, DATE_COL],
                        "entry_date": df.loc[entry_idx, DATE_COL],
                        "exit_date": df.loc[exit_idx, DATE_COL],
                        "position": "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "initial_stop_price": float(initial_stop),
                        "final_stop_price": float(initial_stop),
                        "R": float(R),
                        "pnl": float(pnl),
                        "square_type": sq_type,
                    })

                    i = exit_idx
                    continue

        # ============ LONG ==============
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df, i, DATE_COL, CLOSE_COL,
                slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )

            if sq_idx is not None and sq_idx < n - 2:

                if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:

                    signal_idx = sq_idx + 1
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1

                    entry_price = float(df.loc[entry_idx, CLOSE_COL])
                    exit_price = float(df.loc[exit_idx, CLOSE_COL])

                    initial_stop = df.loc[sq_idx, LOW_COL] - 2 * df.loc[sq_idx, "ATR"]
                    risk = entry_price - initial_stop
                    pnl = exit_price - entry_price
                    R = pnl / risk if risk != 0 else 0.0

                    pts_Tm1 = calc_tminus1_profit(df, signal_idx, "long")
                    pts = calc_forward_point_profits(df, entry_idx, entry_price, "long")

                    trades.append({
                        "trade_no": len(trades) + 1,
                        "square_date": df.loc[sq_idx, DATE_COL],
                        "signal_date": df.loc[signal_idx, DATE_COL],
                        "entry_date": df.loc[entry_idx, DATE_COL],
                        "exit_date": df.loc[exit_idx, DATE_COL],
                        "position": "long",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "initial_stop_price": float(initial_stop),
                        "final_stop_price": float(initial_stop),
                        "R": float(R),
                        "pnl": float(pnl),
                        "square_type": sq_type,
                    })

                    i = exit_idx
                    continue

    # ===== Build DF =====
    trades_df = pd.DataFrame(trades)

    # ===== Equity =====
    df["equity"] = np.nan
    equity = 1.0
    for t in trades:
        equity += equity * RISK_PER_TRADE * t["R"]
        df.loc[df[DATE_COL] >= t["exit_date"], "equity"] = equity

    return trades_df, df
# ==========================
# METRICS + COMMENTARY
# ==========================

def compute_metrics(trades_df, price_df):
    if trades_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0,
            "avg_R": 0,
            "cagr": 0,
            "max_dd": 0,
            "start_date": None,
            "end_date": None,
            "years": 0,
        }

    n = len(trades_df)
    wins = (trades_df["R"] > 0).sum()
    win_rate = 100 * wins / n
    avg_R = trades_df["R"].mean()

    eq = price_df["equity"].dropna()
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]

    start_date = price_df[DATE_COL].iloc[0]
    end_date = price_df[DATE_COL].iloc[-1]
    years = (end_date - start_date).days / 365.25

    if years > 0 and start_eq > 0:
        cagr = (end_eq / start_eq) ** (1 / years) - 1
    else:
        cagr = 0

    equity = eq.values
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    max_dd = float(dd.min()) if len(dd) else 0

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "cagr": cagr,
        "max_dd": max_dd,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
    }
# ==========================
# HTML RENDERING
# ==========================

def render_stock_html(symbol, metrics, trades_df, commentary=""):
    start = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    yrs = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} – Gann T+1 Close System</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 900px;
  margin: auto;
  padding: 20px;
  background: #fafafa;
}}
.card {{
  background: white;
  padding: 16px;
  border-radius: 10px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
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

<h1>{symbol} – Gann T+1 Close System</h1>

<div class="card">
<h2>Summary</h2>
<p><b>Total Trades:</b> {metrics["n_trades"]}<br>
<b>Win Rate:</b> {metrics["win_rate"]:.1f}%<br>
<b>Avg R:</b> {metrics["avg_R"]:.2f}<br>
<b>CAGR:</b> {metrics["cagr"]*100:.1f}%<br>
<b>Max DD:</b> {metrics["max_dd"]*100:.1f}%<br>
<b>Period:</b> {start} to {end} ({yrs} yrs)</p>
</div>

<div class="card">
<h2>Trades</h2>
<table>
<tr>
<th>#</th>
<th>Square</th>
<th>Signal</th>
<th>Entry (Close)</th>
<th>Exit (T+1 Close)</th>
<th>Side</th>
<th>R</th>
<th>SqType</th>
</tr>
"""

    for _, tr in trades_df.iterrows():

        html += f"""
<tr>
<td>{int(tr['trade_no'])}</td>
<td>{tr['square_date'].strftime('%Y-%m-%d')}</td>
<td>{tr['signal_date'].strftime('%Y-%m-%d')}</td>
<td>{tr['entry_date'].strftime('%Y-%m-%d')}</td>
<td>{tr['exit_date'].strftime('%Y-%m-%d')}</td>
<td>{tr['position']}</td>
<td>{tr['R']:.2f}</td>
<td>{tr['square_type']}</td>
</tr>
"""

    html += """
</table>
</div>

</body>
</html>
"""
    return html
def render_master_index(summaries):
    rows = ""
    for s in summaries:
        rows += f"""
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

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NSE Gann T+1 System</title>
<style>
body {{
  font-family: Arial, sans-serif;
  max-width: 1000px;
  margin: auto;
  padding: 20px;
  background: #fafafa;
}}
.card {{
  background: white;
  padding: 16px;
  border-radius: 10px;
  margin-bottom: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
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

<h1>NSE Stocks – Gann T+1 Close Backtests</h1>

<div class="card">
<table>
<tr>
<th>Symbol</th>
<th># Trades</th>
<th>Win%</th>
<th>Avg R</th>
<th>CAGR</th>
<th>Max DD</th>
<th>Years</th>
</tr>
{rows}
</table>
</div>

</body>
</html>
"""
# ==========================
# MAIN
# ==========================

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

        trades_df, price_df = backtest_symbol(df)

        # save CSV
        out_csv = os.path.join(TRADES_CSV_DIR, f"{symbol}_trades.csv")
        trades_df.to_csv(out_csv, index=False)

        # metrics
        metrics = compute_metrics(trades_df, price_df)

        # HTML page
        sym_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(sym_dir, exist_ok=True)

        out_html = os.path.join(sym_dir, "index.html")
        html = render_stock_html(symbol, metrics, trades_df)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        summaries.append({
            "symbol": symbol,
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "avg_R": metrics["avg_R"],
            "cagr": metrics["cagr"],
            "max_dd": metrics["max_dd"],
            "years": metrics["years"],
            "link": f"stocks/{symbol}/index.html",
        })

    # master index
    master = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master)

    print("Site built successfully.")


if __name__ == "__main__":
    main()

