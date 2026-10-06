"""Sharpe 動能輪動的投組回測（與 src.strategy.momentum 同規則），對標 VOO / 0050。

月調（每 21 個交易日）、持有前 TOP_N[市場]、跌出前 KEEP_N[市場] 才賣、每換一檔扣成本 COST。
以「前一日」的分數決定隔日持股，避免偷看未來。結果分全期與前後兩段（樣本外檢查）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategy.momentum import (ALLOCATION, KEEP_N, LOOKBACK, LOWENTRY_DD, LOWENTRY_LT_YEARS, LOWENTRY_SLOTS, TOP_N,
                                   download_closes, universe)

REBALANCE_DAYS = 21
COST = 0.002
BENCHMARK = {"us": "VOO", "tw": "0050.TW"}
# 額外對照：科技 ETF（美股）＋「同一股票池全部等權持有」——看策略是否只是在吃族群本身的漲幅。
EXTRA_COMPARE = {"us": ["QQQ"], "tw": []}
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


def run_backtest(market: str, closes: pd.DataFrame, bench: pd.Series, start: str = "2017-07-01",
                 extra: Optional[Dict[str, pd.Series]] = None, opens: Optional[pd.DataFrame] = None) -> Dict:
    """時序（point-in-time）：T-1 收盤算分數 → T 開盤成交 → 持有報酬以開盤對開盤計算。
    2026-10-07 修正：舊版用 T-1 收盤的分數、又從同一個 T-1 收盤開始計報酬（等於看完收盤價再用收盤價成交）。
    未提供 opens 時退而用「T 收盤成交」（訊號後下一個收盤，仍不使用訊號當日的價格成交）。"""
    closes = closes.sort_index().ffill(limit=3)
    rets = closes.pct_change()
    if opens is not None:
        opens = opens.reindex(closes.index).ffill(limit=3)
        hold_ret = (opens.shift(-1) / opens - 1)          # 第 i 天開盤買進，持有到第 i+1 天開盤
    else:
        hold_ret = rets.shift(-1)                          # 第 i 天收盤買進，賺第 i+1 天的報酬
    score = rets.rolling(LOOKBACK).mean() / rets.rolling(LOOKBACK).std()
    idx = closes.index[closes.index >= start]
    closes, rets, score = closes.loc[idx], rets.loc[idx].fillna(0.0), score.loc[idx]
    hold_ret = hold_ret.loc[idx].fillna(0.0)
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
            new = [k for k in s.index if k not in stay][: max(0, TOP_N[market] - len(stay))]
            nxt = stay + new
            changed = set(nxt) ^ set(held)
            trades += len(changed)
            cost = COST * len(changed) / TOP_N[market]
            if changed:
                log.append({"date": idx[i].strftime("%Y-%m-%d"),
                            "buy": sorted(set(nxt) - set(held)), "sell": sorted(set(held) - set(nxt))})
            held = nxt
        day = float(hold_ret.iloc[i][held].mean()) if held else 0.0
        eq.append(eq[-1] * (1 + day - cost))
    eq_s = pd.Series(eq, index=idx)
    ew = (1 + rets.mean(axis=1).fillna(0.0)).cumprod()   # 同池等權（每日再平衡近似）
    bm = bench / bench.iloc[0]

    periods = {"全期": (None, None), "2017-2021": (None, "2021-12-31"), "2022-今": (SPLIT_DATE, None)}
    out_periods = {}
    for lab, (a, z) in periods.items():
        e, b = eq_s[a:z], bm[a:z]
        if len(e) > 30:
            out_periods[lab] = {"strategy": _stats(e), "benchmark": _stats(b)}
            comp = {"同池等權持有": _stats(ew[a:z])}
            for name, ser in (extra or {}).items():
                comp[name] = _stats(ser.reindex(idx).ffill()[a:z])
            out_periods[lab]["compare"] = comp

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
        "_eq_daily": eq_s,
        "current_holdings": held,
        "recent_rebalances": log[-6:],
        "rules": {"lookback_days": LOOKBACK, "top_n": TOP_N[market], "keep_n": KEEP_N[market],
                  "rebalance_days": REBALANCE_DAYS, "cost_per_trade": COST},
        "caveat": "AI 科技股池是用現在眼光挑出的贏家族群，有明顯倖存者／後見之明偏差：實際報酬會比回測低很多，AI 族群轉弱時也可能大幅落後大盤。",
    }


def lowentry_returns(closes: pd.DataFrame, start: str = "2017-07-01", hold: int = 252,
                     opens: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, List[Dict]]:
    """長線低檔布局：T-1 收盤符合（3 年報酬 > 0 且距 52 週高點 ≤ -30%）→ T 成交，持有 hold 日。
    提供 opens 時：T 開盤成交、開盤對開盤計報酬（與動能回測同一時間基準，組合時才不會虛增分散效果）；
    否則 T 收盤成交、收盤對收盤。每檔 1/LOWENTRY_SLOTS 權重、空槽為現金；同檔出場後 60 日內不重複進場。"""
    closes = closes.sort_index().ffill(limit=3)
    if opens is not None:
        px_exec = opens.reindex(closes.index).ffill(limit=3)
        rets = (px_exec / px_exec.shift(1) - 1)            # 第 j 天的報酬 = 開盤(j-1) → 開盤(j)
    else:
        px_exec = closes
        rets = closes.pct_change()
    dd = closes / closes.rolling(252, min_periods=200).max() - 1
    lt = closes / closes.shift(252 * LOWENTRY_LT_YEARS) - 1
    ok = (dd <= LOWENTRY_DD) & (lt > 0)
    idx = closes.index[closes.index >= start]
    pos: Dict[str, Tuple[int, float]] = {}
    cool: Dict[str, int] = {}
    daily, trades = [], []
    for j in range(1, len(idx)):
        d, prev = idx[j], idx[j - 1]
        r = sum(float(rets.at[d, k]) for k in pos if np.isfinite(rets.at[d, k])) / LOWENTRY_SLOTS
        cost = 0.0
        for k in list(pos):
            j0, p0 = pos[k]
            if j - j0 >= hold:
                trades.append({"ticker": k, "entry": idx[j0].strftime("%Y-%m-%d"), "exit": d.strftime("%Y-%m-%d"),
                               "return_pct": round((px_exec.at[d, k] / p0 - 1) * 100, 1)})
                del pos[k]
                cool[k] = j + 60
                cost += COST / LOWENTRY_SLOTS
        if len(pos) < LOWENTRY_SLOTS:
            row = ok.loc[prev]
            cands = sorted([k for k in row[row].index if k not in pos and cool.get(k, 0) <= j], key=lambda k: dd.at[prev, k])
            for k in cands[: LOWENTRY_SLOTS - len(pos)]:
                if np.isfinite(px_exec.at[d, k]):
                    pos[k] = (j, float(px_exec.at[d, k]))
                    cost += COST / LOWENTRY_SLOTS
        daily.append((d, r - cost))
    ser = pd.Series([x for _, x in daily], index=[d for d, _ in daily])
    return ser, trades


def _period_stats(eq: pd.Series, bench: pd.Series) -> Dict:
    out = {}
    for lab, (a, z) in {"全期": (None, None), "2017-2021": (None, "2021-12-31"), "2022-今": (SPLIT_DATE, None)}.items():
        e, b = eq[a:z], bench[a:z]
        if len(e) > 30:
            out[lab] = {"strategy": _stats(e), "benchmark": _stats(b)}
    return out


def build_backtest_report(period: str = "max") -> Dict:
    out = {}
    for m in ("us", "tw"):
        syms = universe(m)
        pm = download_closes(syms + [BENCHMARK[m]], period=period)
        extra_syms = EXTRA_COMPARE[m]
        if extra_syms:
            pm.update(download_closes(extra_syms, period=period))
        extra = {e: pm.pop(e)["Close"] for e in extra_syms if e in pm}
        bench = pm.pop(BENCHMARK[m], None)
        if bench is None or not pm:
            continue
        closes = pd.DataFrame({s: f["Close"] for s, f in pm.items()})
        closes = closes[closes.index >= "2013-01-01"]
        opens = pd.DataFrame({s: f["Open"] for s, f in pm.items() if "Open" in f}).reindex(closes.index)
        res = run_backtest(m, closes, bench["Close"], extra=extra, opens=opens)
        res["execution_timing"] = "兩條策略皆為 T-1 收盤訊號 → T 開盤成交，開盤對開盤計報酬；組合時兩者對齊同一時間區段"
        # 長線低檔布局與 70/30 組合（使用者設定的配置）
        low_r, low_trades = lowentry_returns(closes, opens=opens)
        eq_m = res.pop("_eq_daily")   # 必須用「每日」動能報酬組合，週取樣會讓波動/Sharpe 失真
        eq_m.index = pd.to_datetime(eq_m.index)
        # 動能 eq_m 第 i 筆 = 開盤(i) → 開盤(i+1)；低檔 low_r 第 j 筆 = 開盤(j-1) → 開盤(j)：動能往後平移一格（shift(1)）對齊同一段時間
        mom_r = eq_m.pct_change().shift(1).reindex(low_r.index).fillna(0)
        combo_r = ALLOCATION["lowentry"] * low_r + ALLOCATION["momentum"] * mom_r
        low_eq, combo_eq = (1 + low_r).cumprod(), (1 + combo_r).cumprod()
        b = bench["Close"].copy()
        b.index = pd.to_datetime(b.index).tz_localize(None) if pd.to_datetime(b.index).tz is not None else pd.to_datetime(b.index)
        b = b.reindex(low_r.index).ffill()
        res["tracks"] = {
            "lowentry": {"name": "長線低檔布局", "periods": _period_stats(low_eq, b),
                         "recent_trades": low_trades[-8:], "trades": len(low_trades),
                         "win_rate_pct": round(sum(1 for t in low_trades if t["return_pct"] > 0) / max(len(low_trades), 1) * 100, 1)},
            "combo": {"name": f"組合：低檔 {ALLOCATION['lowentry']:.0%} ＋ 動能 {ALLOCATION['momentum']:.0%}",
                      "periods": _period_stats(combo_eq, b)},
        }
        weekly = combo_eq.iloc[::5]
        low_w = low_eq.reindex(weekly.index)
        curve = {p["date"]: p for p in res["equity_curve"]}
        res["combo_curve"] = [{"date": d.strftime("%Y-%m-%d"), "combo": round(float(v), 4), "lowentry": round(float(low_w.loc[d]), 4),
                               "benchmark": round(float(b.loc[d] / b.iloc[0]), 4)} for d, v in weekly.items()]
        out[m] = res
    from datetime import datetime, timedelta, timezone
    return {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "markets": out}
