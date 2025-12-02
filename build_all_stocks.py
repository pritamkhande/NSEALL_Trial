def backtest_symbol(df):
    equity = 1.0
    in_trade = False
    position = None
    entry_idx = None
    entry_price = None
    initial_stop_price = None
    entry_square_type = None
    square_idx = None    # store square bar index (setup)

    trades = []
    n = len(df)
    i = 0

    # ================= ORIGINAL SIGNAL ENGINE =================
    while i < n - 3:    # need sq_idx, sq_idx+1, sq_idx+2
        if not in_trade:

            # -------- SHORT SIGNAL --------
            if df.loc[i, "swing_low"]:
                sq_idx, sq_type = find_square_from_swing_low(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 2:

                    # ORIGINAL BREAKOUT: next day close < square low
                    if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:

                        position = "short"
                        square_idx = sq_idx
                        entry_square_type = sq_type

                        # REAL SIGNAL DAY = sq_idx + 1
                        signal_idx = sq_idx + 1
                        signal_date = df.loc[signal_idx, DATE_COL]

                        # ENTRY = today's close (signal day close)
                        entry_idx = signal_idx
                        entry_price = float(df.loc[entry_idx, CLOSE_COL])

                        # EXIT = next day close
                        exit_idx = signal_idx + 1
                        exit_date = df.loc[exit_idx, DATE_COL]
                        exit_price = float(df.loc[exit_idx, CLOSE_COL])

                        # risk using same theoretical SL as original
                        initial_stop = df.loc[sq_idx, HIGH_COL] + 2 * df.loc[sq_idx, "ATR"]

                        risk = initial_stop - entry_price
                        pnl = entry_price - exit_price
                        R = pnl / risk if risk != 0 else 0.0

                        # forward performance
                        pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                        pts = calc_forward_point_profits(df, entry_idx, entry_price, position)

                        trades.append({
                            "trade_no": len(trades) + 1,
                            "square_date": df.loc[square_idx, DATE_COL],
                            "signal_date": signal_date,
                            "entry_date": df.loc[entry_idx, DATE_COL],
                            "exit_date": exit_date,
                            "entry_index": entry_idx,
                            "exit_index": exit_idx,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "initial_stop_price": float(initial_stop),
                            "final_stop_price": float(initial_stop),
                            "R": float(R),
                            "pnl": float(pnl),
                            "exit_reason": "T+1_close",
                            "square_type": entry_square_type,
                            "pts_Tm1": pts_Tm1,
                            "pts_T": pts[0],
                            "pts_T1": pts[1],
                            "pts_T2": pts[2],
                            "pts_T3": pts[3],
                            "pts_T4": pts[4],
                        })

                        i = exit_idx
                        continue

            # -------- LONG SIGNAL --------
            if df.loc[i, "swing_high"]:
                sq_idx, sq_type = find_square_from_swing_high(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 2:

                    if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:

                        position = "long"
                        square_idx = sq_idx
                        entry_square_type = sq_type

                        # REAL SIGNAL DAY = sq_idx + 1
                        signal_idx = sq_idx + 1
                        signal_date = df.loc[signal_idx, DATE_COL]

                        # ENTRY = today's close
                        entry_idx = signal_idx
                        entry_price = float(df.loc[entry_idx, CLOSE_COL])

                        # EXIT = next day close
                        exit_idx = signal_idx + 1
                        exit_date = df.loc[exit_idx, DATE_COL]
                        exit_price = float(df.loc[exit_idx, CLOSE_COL])

                        initial_stop = df.loc[sq_idx, LOW_COL] - 2 * df.loc[sq_idx, "ATR"]

                        risk = entry_price - initial_stop
                        pnl = exit_price - entry_price
                        R = pnl / risk if risk != 0 else 0.0

                        pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                        pts = calc_forward_point_profits(df, entry_idx, entry_price, position)

                        trades.append({
                            "trade_no": len(trades) + 1,
                            "square_date": df.loc[square_idx, DATE_COL],
                            "signal_date": signal_date,
                            "entry_date": df.loc[entry_idx, DATE_COL],
                            "exit_date": exit_date,
                            "entry_index": entry_idx,
                            "exit_index": exit_idx,
                            "position": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "initial_stop_price": float(initial_stop),
                            "final_stop_price": float(initial_stop),
                            "R": float(R),
                            "pnl": float(pnl),
                            "exit_reason": "T+1_close",
                            "square_type": entry_square_type,
                            "pts_Tm1": pts_Tm1,
                            "pts_T": pts[0],
                            "pts_T1": pts[1],
                            "pts_T2": pts[2],
                            "pts_T3": pts[3],
                            "pts_T4": pts[4],
                        })

                        i = exit_idx
                        continue

            i += 1

    # Build DataFrame and equity curve
    trades_df = pd.DataFrame(trades)

    df["equity"] = np.nan
    equity = 1.0
    it = iter(trades)
    ct = next(it, None)

    for idx in range(len(df)):
        d = df.loc[idx, DATE_COL]
        while ct is not None and ct["exit_date"] <= d:
            r = ct["R"]
            equity += (equity * RISK_PER_TRADE * r)
            ct = next(it, None)
        df.loc[idx, "equity"] = equity

    return trades_df, df
