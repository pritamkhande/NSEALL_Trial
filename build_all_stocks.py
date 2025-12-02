import os
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD"                 # folder with subfolders A, B, ..., 0-9
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
    equity = 1.0
    in_trade = False
    position = None
    entry_idx = None
    entry_price = None
    stop_price = None
    initial_stop_price = None
    entry_square_type = None
    signal_idx = None
    signal_date = None

    trades = []
    n = len(df)
    i = 0

    # ================= ORIGINAL SIGNAL + EXIT ENGINE =================
    # This block is your old logic
    # (entry at next-day OPEN, ATR trailing exits).
    while i < n - 2:
        if not in_trade:

            # SHORT from swing low
            if df.loc[i, "swing_low"]:
                sq_idx, sq_type = find_square_from_swing_low(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 1:
                    # breakout: next-day close below square low
                    if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:
                        in_trade = True
                        position = "short"
                        entry_idx = sq_idx + 1              # breakout day
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type

                        sl = df.loc[sq_idx, HIGH_COL] + 2 * df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        initial_stop_price = sl
                        signal_idx = sq_idx                  # square bar index
                        signal_date = df.loc[sq_idx, DATE_COL]
                        i = entry_idx
                        continue

            # LONG from swing high
            if df.loc[i, "swing_high"]:
                sq_idx, sq_type = find_square_from_swing_high(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 1:
                    # breakout: next-day close above square high
                    if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:
                        in_trade = True
                        position = "long"
                        entry_idx = sq_idx + 1              # breakout day
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type

                        sl = df.loc[sq_idx, LOW_COL] - 2 * df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        initial_stop_price = sl
                        signal_idx = sq_idx
                        signal_date = df.loc[sq_idx, DATE_COL]
                        i = entry_idx
                        continue

            i += 1

        else:
            atr = df.loc[i, "ATR"]
            close = df.loc[i, CLOSE_COL]
            high = df.loc[i, HIGH_COL]
            low = df.loc[i, LOW_COL]
            date_i = df.loc[i, DATE_COL]

            if position == "long":
                trail = close - 3 * atr
                if trail > stop_price:
                    stop_price = trail
            else:
                trail = close + 3 * atr
                if trail < stop_price:
                    stop_price = trail

            exit_reason = None
            exit_price = None

            if position == "long":
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
            else:
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"

            if i == n - 1 and exit_reason is None:
                exit_price = close
                exit_reason = "End"

            if exit_reason:
                # ORIGINAL R and pnl (will be recomputed later)
                if position == "long":
                    risk = entry_price - initial_stop_price
                    pnl = exit_price - entry_price
                else:
                    risk = initial_stop_price - entry_price
                    pnl = entry_price - exit_price

                r_mult = pnl / risk if risk != 0 else 0

                pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                pts = calc_forward_point_profits(df, entry_idx, entry_price, position, max_horizon=4)

                trades.append({
                    "trade_no": len(trades) + 1,
                    "signal_index": signal_idx,        # square bar index
                    "signal_date": signal_date,        # square bar date
                    "entry_index": entry_idx,          # breakout bar index
                    "exit_index": i,                   # original exit index
                    "entry_date": df.loc[entry_idx, DATE_COL],
                    "exit_date": date_i,
                    "position": position,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "initial_stop_price": float(initial_stop_price),
                    "final_stop_price": float(stop_price),
                    "R": float(r_mult),
                    "pnl": float(pnl),
                    "exit_reason": exit_reason,
                    "square_type": entry_square_type,
                    "pts_Tm1": pts_Tm1,
                    "pts_T": pts[0],
                    "pts_T1": pts[1],
                    "pts_T2": pts[2],
                    "pts_T3": pts[3],
                    "pts_T4": pts[4],
                })

                risk_amt = equity * RISK_PER_TRADE
                equity += r_mult * risk_amt

                in_trade = False
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                initial_stop_price = None
                entry_square_type = None
                signal_idx = None
                signal_date = None

            i += 1

    # ================= RE-MAP TO: ENTRY = BREAKOUT CLOSE, EXIT = NEXT CLOSE =================
    #
    # Now we keep EXACTLY the same trades (same signals, same count).
    # Only change:
    #   entry_index_new = original entry_index  (breakout day)
    #   exit_index_new  = entry_index_new + 1   (next day)
    #   entry_price_new = Close[entry_index_new]
    #   exit_price_new  = Close[exit_index_new]
    #
    # This matches: "enter at closing price on signal/breakout day, exit next day close".

    for t in trades:
        pos = t["position"]

        entry_idx_orig = t["entry_index"]     # this is breakout bar index
        entry_idx_new = entry_idx_orig
        exit_idx_new = entry_idx_new + 1      # next day close

        # guard
        if exit_idx_new >= len(df):
            # cannot define T+1 exit for the last bar, keep original exit
            continue

        # indices & dates
        t["entry_index"] = entry_idx_new
        t["exit_index"] = exit_idx_new
        t["entry_date"] = df.loc[entry_idx_new, DATE_COL]
        t["exit_date"] = df.loc[exit_idx_new, DATE_COL]

        # prices
        entry_price_new = float(df.loc[entry_idx_new, CLOSE_COL])
        exit_price_new = float(df.loc[exit_idx_new, CLOSE_COL])
        t["entry_price"] = entry_price_new
        t["exit_price"] = exit_price_new

        # risk with same theoretical SL
        initial_stop = float(t["initial_stop_price"])

        if pos == "long":
            risk = entry_price_new - initial_stop
            pnl = exit_price_new - entry_price_new
        else:
            risk = initial_stop - entry_price_new
            pnl = entry_price_new - exit_price_new

        t["pnl"] = float(pnl)
        t["R"] = float(pnl / risk) if risk != 0 else 0.0
        t["final_stop_price"] = initial_stop
        t["exit_reason"] = "T+1_close"

        # recompute T(-1) and forward P&L from this new entry
        sig_idx = t["signal_index"]
        t["pts_Tm1"] = calc_tminus1_profit(df, sig_idx, pos)
        pts = calc_forward_point_profits(df, entry_idx_new, entry_price_new, pos, max_horizon=4)
        t["pts_T"] = pts[0]
        t["pts_T1"] = pts[1]
        t["pts_T2"] = pts[2]
        t["pts_T3"] = pts[3]
        t["pts_T4"] = pts[4]

    trades_df = pd.DataFrame(trades)

    # ================= RECOMPUTE EQUITY WITH NEW R =================
    df["equity"] = np.nan
    equity = 1.0
    it = iter(trades)
    ct = next(it, None)

    for idx in range(n):
        d = df.loc[idx, DATE_COL]
        while ct is not None and ct["exit_date"] <= d:
            r = ct["R"]
            ra = equity * RISK_PER_TRADE
            equity += r * ra
            ct = next(it, None)
        df.loc[idx, "equity"] = equity

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


def build_system_commentary(symbol, metrics, trades_df):
    n = metrics["n_trades"]
    years = metrics["years"]
    avg_R = metrics["avg_R"]
    win_rate = metrics["win_rate"]
    cagr = metrics["cagr"] * 100
    max_dd = metrics["max_dd"] * 100

    if years > 0:
        trades_per_year = n / years
    else:
        trades_per_year = 0

    if trades_df.empty:
        return f"No trades for {symbol}. Current parameters too strict."

    avg_hold = (trades_df["exit_index"] - trades_df["entry_index"]).mean()

    style = []
    if trades_per_year < 5:
        style.append("very selective system")
    elif trades_per_year < 15:
        style.append("moderately active system")
    else:
        style.append("high-frequency swing system")

    if max_dd < 5:
        style.append("very low risk")
    elif max_dd < 12:
        style.append("moderate risk")
    else:
        style.append("high risk")

    if cagr < 2:
        style.append("research oriented")
    elif cagr < 8:
        style.append("balanced return profile")
    else:
        style.append("aggressive return profile")

    desc = ", ".join(style)

    return (
        f"For {symbol}, total {n} trades over {years:.1f} years, "
        f"avg hold ~{avg_hold:.1f} bars, win rate {win_rate:.1f}%, "
        f"avg {avg_R:.2f}R, CAGR ~{cagr:.1f}%, max DD ~{max_dd:.1f}%. "
        f"Overall: {desc}."
    )


# ==========================
# HTML RENDERING
# ==========================

def render_stock_html(symbol, metrics, trades_df, commentary):
    start = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    yrs = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{symbol} – Gann Squaring</title>
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

<h1>{symbol} – Gann Squaring Backtest</h1>

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
<h2>System Commentary</h2>
<p>{commentary}</p>
</div>

<div class="card">
<h2>All Trades</h2>
<table>
<tr>
<th>#</th>
<th>Signal (square date)</th>
<th>Entry (breakout close)</th>
<th>Exit (next-day close)</th>
<th>Side</th>
<th>R</th>
<th>Sq-Type</th>
<th>ExitReason</th>
<th>T(-1)</th>
<th>T</th>
<th>T+1</th>
<th>T+2</th>
<th>T+3</th>
<th>T+4</th>
</tr>
"""

    for _, tr in trades_df.iterrows():
        sig = tr["signal_date"].strftime("%Y-%m-%d") if pd.notna(tr["signal_date"]) else "NA"
        ent = tr["entry_date"].strftime("%Y-%m-%d")
        ext = tr["exit_date"].strftime("%Y-%m-%d")

        html += f"""
<tr>
<td>{int(tr['trade_no'])}</td>
<td>{sig}</td>
<td>{ent}</td>
<td>{ext}</td>
<td>{tr['position']}</td>
<td>{tr['R']:.2f}</td>
<td>{tr['square_type']}</td>
<td>{tr['exit_reason']}</td>
<td>{tr['pts_Tm1']:.2f}</td>
<td>{tr['pts_T']:.2f}</td>
<td>{tr['pts_T1']:.2f}</td>
<td>{tr['pts_T2']:.2f}</td>
<td>{tr['pts_T3']:.2f}</td>
<td>{tr['pts_T4']:.2f}</td>
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

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NSE Stocks – Gann System</title>
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

<h1>NSE Stocks – Gann Squaring Backtests</h1>

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

        early_df = load_early_close_for_symbol(symbol)
        if early_df is not None and not trades_df.empty:
            # currently no early-close adjustment
            pass

        out_csv = os.path.join(TRADES_CSV_DIR, f"{symbol}_trades.csv")
        trades_df.to_csv(out_csv, index=False)

        metrics = compute_metrics(trades_df, price_df)
        commentary = build_system_commentary(symbol, metrics, trades_df)

        sym_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(sym_dir, exist_ok=True)

        out_html = os.path.join(sym_dir, "index.html")
        html = render_stock_html(symbol, metrics, trades_df, commentary)
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

    master = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master)

    print("Site built successfully.")


if __name__ == "__main__":
    main()
