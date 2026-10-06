"""Sharpe 動能輪動的投組回測（與 src.strategy.momentum 同規則），對標 VOO / 0050。

月調（每 21 個交易日）、持有前 TOP_N、跌出前 KEEP_N[市場] 才賣、每換一檔扣成本 COST。
以「前一日」的分數決定隔日持股，避免偷看未來。結果分全期與前後兩段（樣本外檢查）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategy.momentum import KEEP_N, LOOKBACK, TOP_N, download_closes, universe

REBALANCE_DAYS = 21
COST = 0.002
BENCHMARK = {"us": "VOO", "tw": "0050.TW"}
SPLIT_DATE = "2022-01-01"


def _stats(eq: pd.Series) -> Dict:
    eq = eq / eq.iloc[0]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dr = eq.pct_change().dropna()
    return {
        "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
        "sharpe": round(float(dr.mean() / dr.std() * math.sqrt(252)), 2) if dr.std() else 0.0,
    }


def run_backtest(market: str, closes: pd.DataFrame, bench: pd.Series, start: str = "2017-07-01") -> Dict:
    closes = closes.sort_index().ffill(limit=3)
    rets = closes.pct_change()
    score = rets.rolling(LOOKBACK).mean() / rets.rolling(LOOKBACK).std()
    idx = closes.index[closes.index >= start]
    closes, rets, score = closes.loc[idx], rets.loc[idx].fillna(0.0), score.loc[idx]
    bench = bench.reindex(idx).ffill()

    held: List[str] = []
    eq = [1.0]
    trades = 0
    log: List[Dict] = []
    for i in range(1, len(idx)):
        cost = 0.0
        if (i - 1) % REBALANCE_DAYS == 0:
            s = score.iloc[i - 1].dropna().sort_values(ascending=False)
            pos = {k: j for j, k in enumerate(s.index)}
            stay = [h for h in held if pos.get(h, 10 ** 9) < KEEP_N[market]]
            new = [k for k in s.index if k not in stay][: max(0, TOP_N - len(stay))]
            nxt = stay + new
            changed = set(nxt) ^ set(held)
            trades += len(changed)
            cost = COST * len(changed) / TOP_N
            if changed:
                log.append({"date": idx[i].strftime("%Y-%m-%d"),
                            "buy": sorted(set(nxt) - set(held)), "sell": sorted(set(held) - set(nxt))})
            held = nxt
        day = float(rets.iloc[i][held].mean()) if held else 0.0
        eq.append(eq[-1] * (1 + day - cost))
    eq_s = pd.Series(eq, index=idx)
    bm = bench / bench.iloc[0]

    periods = {"全期": (None, None), "2017-2021": (None, "2021-12-31"), "2022-今": (SPLIT_DATE, None)}
    out_periods = {}
    for lab, (a, z) in periods.items():
        e, b = eq_s[a:z], bm[a:z]
        if len(e) > 30:
            out_periods[lab] = {"strategy": _stats(e), "benchmark": _stats(b)}

    weekly = eq_s.iloc[::5]
    bm_weekly = bm.reindex(weekly.index)
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "market": market,
        "benchmark_symbol": BENCHMARK[market],
        "start_date": idx[0].strftime("%Y-%m-%d"),
        "end_date": idx[-1].strftime("%Y-%m-%d"),
        "universe_size": int(closes.shape[1]),
        "trades_per_year": round(trades / years, 1),
        "periods": out_periods,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "strategy": round(float(v), 4), "benchmark": round(float(bm_weekly.loc[d]), 4)}
            for d, v in weekly.items()
        ],
        "current_holdings": held,
        "recent_rebalances": log[-6:],
        "rules": {"lookback_days": LOOKBACK, "top_n": TOP_N, "keep_n": KEEP_N[market],
                  "rebalance_days": REBALANCE_DAYS, "cost_per_trade": COST},
        "caveat": "股票池為現在的成分股（含事後挑選的 AI 主線股），有倖存者偏差，實際超額報酬會比回測小。",
    }


def build_backtest_report(period: str = "10y") -> Dict:
    out = {}
    for m in ("us", "tw"):
        syms = universe(m)
        pm = download_closes(syms + [BENCHMARK[m]], period=period)
        bench = pm.pop(BENCHMARK[m], None)
        if bench is None or not pm:
            continue
        closes = pd.DataFrame({s: f["Close"] for s, f in pm.items()})
        out[m] = run_backtest(m, closes, bench["Close"])
    from datetime import datetime, timedelta, timezone
    return {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "markets": out}
