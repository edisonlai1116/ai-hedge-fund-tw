"""完整驗證：python -m src.ranking.validate [--market us|tw] [--out docs/data/ranking_validation.json]

所有結果皆為 point-in-time（T-1 收盤分數 → T 開盤成交，open-to-open 報酬，每次換股成本 0.2%）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd

from src.ranking import backtest as bt
from src.ranking import features as fx
from src.ranking.themes import industry_benchmark, theme_of

BENCH = {"us": ["SPY", "QQQ", "SMH", "XLK", "XLU", "XLE", "XLI", "XLC", "XLY", "IGV"], "tw": ["0050.TW"]}
STRATEGY_PARAMS = {"us": (10, 50), "tw": (20, 30)}   # 與網站現行策略相同的持股數/緩衝


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round((c - h) * 100, 1), round((c + h) * 100, 1))


def module_accuracy(panels: Dict[str, pd.DataFrame], close: pd.DataFrame, bench: pd.Series, start: str,
                    horizon: int = 20, step: int = 5, min_n: int = 300) -> Dict:
    """各模組「看多」（≥60）時未來 horizon 日跑贏大盤的比率，含 95% 信賴區間；樣本不足不下結論。"""
    fwd = close.shift(-horizon) / close - 1
    ex = fwd.sub((bench.shift(-horizon) / bench - 1).reindex(close.index), axis=0)
    out = {}
    for name, p in panels.items():
        if name == "sharpe_126":
            continue
        k = n = 0
        for d in p.index[p.index >= start][::step]:
            s, e = p.loc[d], ex.loc[d]
            m = (s >= 60) & e.notna()
            n += int(m.sum())
            k += int((e[m] > 0).sum())
        lo, hi = _wilson(k, n)
        out[name] = {"bullish_signals": n, f"beat_market_{horizon}d_pct": round(k / n * 100, 1) if n else None,
                     "ci95": [lo, hi], "conclusive": bool(n >= min_n and (lo or 0) > 50 or (hi or 100) < 50),
                     "note": "樣本不足，不據此調權重" if n < min_n else ""}
    return out


def regime_exposure(b: Dict[str, pd.Series], idx: pd.DatetimeIndex) -> pd.Series:
    """與 regime.classify_regime 同規則的向量化版本（point-in-time）。"""
    spy = b["SPY"].reindex(idx).ffill()
    ma = spy.rolling(200).mean()
    r20 = spy / spy.shift(20) - 1
    vix = b.get("VIX", pd.Series(index=idx, dtype=float)).reindex(idx).ffill()
    risk_off = (spy < ma) | (vix >= 28) | (r20 <= -0.07)
    exp = pd.Series(0.85, index=idx)
    bull = (spy >= ma) & (vix < 20)
    exp[bull] = 1.0
    exp[risk_off] = 0.5
    return exp.where(ma.notna(), 1.0)


def run(market: str = "us") -> Dict:
    from src.strategy.momentum import download_closes, universe
    syms = universe(market)
    pm = download_closes(syms + BENCH[market] + (["^VIX"] if market == "us" else []), period="max")
    pm = {k: v[v.index >= pd.Timestamp("2015-06-01", tz=v.index.tz) if v.index.tz is not None else v.index >= "2015-06-01"] for k, v in pm.items()}
    bclose = {b: pm.pop(b)["Close"] for b in BENCH[market] if b in pm}
    bopen = {b: pm[b]["Open"] for b in []}
    vix = pm.pop("^VIX", None)
    if market == "tw":
        bclose["SPY"] = bclose["QQQ"] = bclose["0050.TW"]
    themes = {s: (theme_of(s).theme if theme_of(s) else None) for s in pm}
    inds = {s: industry_benchmark(s) for s in pm}
    F = fx.compute_features(pm, bclose, themes, inds)
    P = bt.score_panels(F)
    open_px = fx.panel(pm, "Open").reindex(F["close"].index).ffill(limit=3)
    close = F["close"]
    top, keep = STRATEGY_PARAMS[market]
    bname = "SPY" if market == "us" else "0050.TW"
    bench_open = {}
    for b in (["SPY", "QQQ"] if market == "us" else ["0050.TW"]):
        s = fx._align(bclose[b], close.index)
        bench_open[b] = s

    # v1：最初的多模組價格綜合分數（相對強度/技術/產業動能/低風險 + 過熱扣分）——保留作為消融比較
    v1_w = {"relative_strength": 0.16, "technical": 0.12, "industry_momentum": 0.10, "low_risk": 0.08}
    v1 = lambda ex=(), pen=0.6: bt.opportunity_from(P, exclude=ex, penalty=pen, weights=v1_w,  # noqa: E731
                                                      components=tuple(v1_w))
    strategies = {
        "Sharpe 動能（現行網站策略）": P["sharpe_126"],
        "Opportunity 最終（動能排名）": bt.opportunity_from(P),
        "Opportunity v1（多模組價格綜合）": v1(),
        "  v1 − 相對強度": v1(("relative_strength",)),
        "  v1 − 技術": v1(("technical",)),
        "  v1 − 產業動能": v1(("industry_momentum",)),
        "  v1 − 低風險": v1(("low_risk",)),
        "  v1 − 過熱扣分": v1(pen=0.0),
        "  動能排名 + 產業動能（未採用）": bt.opportunity_from(P, weights={"momentum_rank": 0.55, "industry_momentum": 0.06},
                                                      components=("momentum_rank", "industry_momentum")),
    }
    results, raw = {}, {}
    for name, score in strategies.items():
        res = bt.run_strategy(score, open_px, top, keep)
        raw[name] = res
        results[name] = bt.split_metrics(res, bench_open)
    res = bt.run_strategy(strategies["Opportunity 最終（動能排名）"], open_px, top, keep,
                          overext=P["overextension"], wait_above=60)
    raw["最終策略：動能排名 + 過熱≥60 等降溫才進"] = res
    results["最終策略：動能排名 + 過熱≥60 等降溫才進"] = bt.split_metrics(res, bench_open)
    exp = regime_exposure({"SPY": bclose[bname], **({"VIX": vix["Close"]} if vix is not None else {})}, close.index)
    res = bt.run_strategy(strategies["Opportunity 最終（動能排名）"], open_px, top, keep, exposure=exp)
    results["動能排名 + 市場狀態曝險（未採用）"] = bt.split_metrics(res, bench_open)
    res = bt.run_strategy(P["sharpe_126"], open_px, top, keep, exposure=exp)
    results["Sharpe 動能 + 市場狀態曝險"] = bt.split_metrics(res, bench_open)

    # 比較基準
    start_idx = close.index[close.index >= "2017-07-01"]
    ew = (1 + (open_px.shift(-1) / open_px - 1).loc[start_idx].mean(axis=1).fillna(0)).cumprod()
    results["同池等權持有"] = bt.split_metrics({"equity": ew, "trades": [], "turnover_per_year": 0}, bench_open)
    for b, s in bench_open.items():
        e = s.loc[start_idx]
        results[f"{b} 買進持有"] = bt.split_metrics({"equity": e / e.iloc[0], "trades": [], "turnover_per_year": 0}, bench_open)

    # 參數敏感度（Opportunity 價格版，樣本外 Sharpe）
    sens = []
    for t in ((5, 10, 15) if market == "us" else (10, 20, 30)):
        for k in ((20, 50, 100) if market == "us" else (30, 40)):
            if k < t:
                continue
            for wait in (None, 60, 70):
                res = bt.run_strategy(strategies["Opportunity 最終（動能排名）"], open_px, t, k,
                                      overext=P["overextension"], wait_above=wait)
                mm = bt.split_metrics(res, bench_open)
                i, o_ = mm.get("In-sample 2017-2021", {}), mm.get("Out-of-sample 2022-", {})
                sens.append({"top": t, "keep": k, "wait_above": wait, "IS_CAGR": i.get("CAGR"), "IS_Sharpe": i.get("Sharpe"),
                             "OOS_CAGR": o_.get("CAGR"), "OOS_Sharpe": o_.get("Sharpe"), "OOS_MaxDD": o_.get("MaxDD")})

    states = bt.market_states(fx._align(bclose[bname], close.index))
    final = "最終策略：動能排名 + 過熱≥60 等降溫才進"
    state_view = {n: bt.state_metrics(raw[n]["equity"], states, bench_open[bname])
                  for n in ("Sharpe 動能（現行網站策略）", "Opportunity v1（多模組價格綜合）", final)}
    wf = bt.walk_forward({"sharpe": P["sharpe_126"], "industry_mix": strategies["  動能排名 + 產業動能（未採用）"],
                          "v1": strategies["Opportunity v1（多模組價格綜合）"]}, open_px, top, keep)
    wf_m = bt.metrics(wf["equity"], None, bench_open)
    conc = {n: bt.ticker_concentration(raw[n]["trades"]) for n in ("Sharpe 動能（現行網站策略）", final)}
    acc = module_accuracy({k: v for k, v in P.items() if k != "overextension"}, close, bclose[bname].reindex(close.index).ffill(), bt.OOS_START)
    cal = bt.calibration_table(strategies["Opportunity 最終（動能排名）"], close, bclose[bname].reindex(close.index).ffill())
    return {
        "market": market, "universe_size": int(close.shape[1]),
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "execution_timing": "score at T-1 close → execute T open → open-to-open returns; cost 0.2%/trade",
        "params": {"top": top, "keep": keep, "rebalance_days": 21},
        "results": results, "sensitivity": sens, "market_states": state_view,
        "walk_forward": {"metrics": wf_m, "yearly_picks": wf["picks"]},
        "overfit_check": conc, "module_accuracy_oos": acc, "calibration": cal,
        "caveats": [
            "股票池為今天的 AI 科技股成分（後見之明/倖存者偏差）：所有策略與『同池等權持有』都被高估，比較相對差異較有意義。",
            "基本面、財報加速、估值、新聞催化、AI 曝險沒有 point-in-time 歷史 → 未進回測，其有效性尚未驗證。",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "tw"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = run(a.market)
    path = a.out or os.path.join("docs", "data", f"ranking_validation_{a.market}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("results", "walk_forward", "overfit_check")}, ensure_ascii=False, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
