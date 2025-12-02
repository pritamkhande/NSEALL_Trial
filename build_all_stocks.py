def backtest_symbol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = []
    n = len(df)

    for i in range(len(df) - 2):

        # --- SHORT SIGNAL ---
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )

            if sq_idx is not None and sq_idx < n - 2:
                # ORIGINAL BREAKOUT (same)
                if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:

                    # REAL SIGNAL DAY = sq_idx + 1
                    signal_idx = sq_idx + 1
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1

                    entry_date = df.loc[entry_idx, DATE_COL]
                    exit_date = df.loc[exit_idx, DATE_COL]

                    entry_price = float(df.loc[entry_idx, CLOSE_COL])
                    exit_price = float(df.loc[exit_idx, CLOSE_COL])

                    # SL is theoretical only
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
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "position": "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "initial_stop_price": float(initial_stop),
                        "final_stop_price": float(initial_stop),
                        "R": float(R),
                        "pnl": float(pnl),
                        "exit_reason": "T+1_close",
                        "square_type": sq_type,
                        "pts_Tm1": pts_Tm1,
                        "pts_T": pts[0],
                        "pts_T1": pts[1],
                        "pts_T2": pts[2],
                        "pts_T3": pts[3],
                        "pts_T4": pts[4],
                    })

                    # SKIP FORWARD: prevent overlapping signals
                    i = exit_idx
                    continue

        # --- LONG SIGNAL ---
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )

            if sq_idx is not None and sq_idx < n - 2:
                if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:

                    signal_idx = sq_idx + 1
                    entry_idx = signal_idx
                    exit_idx = signal_idx + 1

                    entry_date = df.loc[entry_idx, DATE_COL]
                    exit_date = df.loc[exit_idx, DATE_COL]

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
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "position": "long",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "initial_stop_price": float(initial_stop),
                        "final_stop_price": float(initial_stop),
                        "R": float(R),
                        "pnl": float(pnl),
                        "exit_reason": "T+1_close",
                        "square_type": sq_type,
                        "pts_Tm1": pts_Tm1,
                        "pts_T": pts[0],
                        "pts_T1": pts[1],
                        "pts_T2": pts[2],
                        "pts_T3": pts[3],
                        "pts_T4": pts[4],
                    })

                    i = exit_idx
                    continue

    # --- Build trades_df ---
    trades_df = pd.DataFrame(trades)

    # --- Build equity curve ---
    df["equity"] = np.nan
    equity = 1.0
    for t in trades:
        equity += equity * RISK_PER_TRADE * t["R"]
        df.loc[df[DATE_COL] >= t["exit_date"], "equity"] = equity

    return trades_df, df
