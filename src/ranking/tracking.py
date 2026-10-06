"""訊號追蹤與校準：每次產生 BUY/SELL/HOLD 都存檔，之後計算實際結果與各模組準確度。

存檔欄位：ticker、date、status、confidence、opportunity、entry_price、各模組分數。
評估：5/20/60 日報酬與相對 SPY 超額、60 日內最大有利/不利波動（MFE/MAE）。
彙整：
  signal_accuracy  各狀態（BUY_NOW、BUY_ON_PULLBACK、SELL…）的 20 日勝率
  analyst_accuracy 各模組「看多（≥60）」20 日跑贏大盤、「看空（≤40）」20 日跑輸大盤的比率
  calibration      機會分數分組 vs 實際 60 日超額報酬
防過度調整：最少樣本 MIN_SAMPLE、只看最近 ROLLING_DAYS 天、附 95% Wilson 信賴區間；
  只有樣本足夠且信賴區間不含 50% 時才標記 conclusive。本模組只報告，不自動改權重。
隱私：只記錄策略觀察名單的公開代號，不含任何持股、成本、股數。
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_SAMPLE = 30
ROLLING_DAYS = 252
HORIZONS = (5, 20, 60)
MODULE_KEYS = ("momentum_rank", "earnings_acceleration", "valuation", "quality", "growth", "catalyst",
               "ai_exposure", "industry_momentum", "relative_strength", "technical")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round((c - h) * 100, 1), round((c + h) * 100, 1))


def log_signals(ranking: Dict, path: str, asof: Optional[str] = None) -> int:
    """把一次 ranking 的每檔結果追加到 jsonl（同一天同一檔只記一次）。"""
    day = asof or date.today().isoformat()
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r["date"], r["ticker"]))
                except Exception:
                    continue
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in ranking.get("ranking", []):
            if r.get("opportunity") is None or (day, r["ticker"]) in seen:
                continue
            rec = {"date": day, "ticker": r["ticker"], "status": r["status"], "flags": r.get("flags", []),
                   "opportunity": r["opportunity"], "confidence": round(abs(r["opportunity"] - 50) * 2),
                   "entry_price": r.get("price"), "regime": ranking.get("regime", {}).get("regime"),
                   "scores": {k: r["scores"].get(k) for k in MODULE_KEYS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _outcomes(recs: List[Dict], closes: Dict[str, pd.Series], spy: pd.Series) -> List[Dict]:
    out = []
    for r in recs:
        c = closes.get(r["ticker"])
        if c is None or c.empty:
            continue
        d0 = pd.Timestamp(r["date"])
        after = c[c.index > d0]
        before = c[c.index <= d0]
        if before.empty or after.empty:
            continue
        p0 = float(before.iloc[-1])
        s_before, s_after = spy[spy.index <= d0], spy[spy.index > d0]
        row = dict(r)
        for h in HORIZONS:
            if len(after) >= h and len(s_after) >= h and len(s_before):
                ret = float(after.iloc[h - 1] / p0 - 1)
                bret = float(s_after.iloc[h - 1] / s_before.iloc[-1] - 1)
                row[f"ret_{h}d"], row[f"excess_{h}d"] = ret, ret - bret
        win = after.iloc[:60]
        if len(win):
            row["mfe_60d"] = float(win.max() / p0 - 1)
            row["mae_60d"] = float(win.min() / p0 - 1)
        out.append(row)
    return out


def evaluate(path: str, today: Optional[date] = None, price_loader=None) -> Dict:
    """讀取訊號紀錄、補上實際結果，輸出準確度彙整。price_loader(symbols)->{sym: close Series}。"""
    today = today or date.today()
    if not os.path.exists(path):
        return {"n_signals": 0, "note": "尚無訊號紀錄"}
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    cutoff = (today - timedelta(days=int(ROLLING_DAYS * 365 / 252))).isoformat()
    recs = [r for r in recs if r["date"] >= cutoff]
    if not recs:
        return {"n_signals": 0, "note": "滾動視窗內無紀錄"}
    if price_loader is None:
        from src.strategy.momentum import download_closes

        def price_loader(symbols):
            pm = download_closes(symbols, period="2y")
            return {k: _naive(v["Close"]) for k, v in pm.items()}
    closes = price_loader(sorted({r["ticker"] for r in recs} | {"SPY"}))
    spy = closes.pop("SPY", pd.Series(dtype=float))
    rows = _outcomes(recs, closes, spy)

    def acc(sel: List[Dict], key: str, want_positive: bool) -> Dict:
        vals = [r[key] for r in sel if key in r]
        n = len(vals)
        k = sum(1 for v in vals if (v > 0) == want_positive)
        lo, hi = wilson(k, n)
        conclusive = n >= MIN_SAMPLE and lo is not None and (lo > 50 or hi < 50)
        return {"n": n, "hit_rate_pct": round(k / n * 100, 1) if n else None, "ci95": [lo, hi],
                "mean_excess_pct": round(float(np.mean(vals)) * 100, 2) if n else None,
                "conclusive": conclusive, "note": "" if n >= MIN_SAMPLE else f"樣本 < {MIN_SAMPLE}，不下結論"}

    signal_acc = {}
    for st in sorted({r["status"] for r in rows}):
        sel = [r for r in rows if r["status"] == st]
        bullish = st in ("BUY_NOW", "BUY_ON_PULLBACK", "HOLD_CORE", "HOLD")
        signal_acc[st] = {f"{h}d": acc(sel, f"excess_{h}d", bullish) for h in HORIZONS}
        mfe = [r["mfe_60d"] for r in sel if "mfe_60d" in r]
        mae = [r["mae_60d"] for r in sel if "mae_60d" in r]
        signal_acc[st]["avg_mfe_60d_pct"] = round(float(np.mean(mfe)) * 100, 1) if mfe else None
        signal_acc[st]["avg_mae_60d_pct"] = round(float(np.mean(mae)) * 100, 1) if mae else None
    module_acc = {}
    for m in MODULE_KEYS:
        bull = [r for r in rows if (r["scores"].get(m) or -1) >= 60]
        bear = [r for r in rows if r["scores"].get(m) is not None and r["scores"][m] <= 40]
        module_acc[m] = {"bullish_20d": acc(bull, "excess_20d", True), "bullish_60d": acc(bull, "excess_60d", True),
                         "bearish_20d": acc(bear, "excess_20d", False)}
    calib = []
    for lo_, hi_ in ((0, 40), (40, 55), (55, 65), (65, 75), (75, 101)):
        sel = [r["excess_60d"] for r in rows if "excess_60d" in r and lo_ <= r["opportunity"] < hi_]
        calib.append({"opportunity": f"{lo_}-{min(hi_, 100)}", "n": len(sel),
                      "mean_excess_60d_pct": round(float(np.mean(sel)) * 100, 2) if sel else None})
    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "n_signals": len(recs),
            "n_with_outcomes": sum(1 for r in rows if "excess_20d" in r), "rolling_days": ROLLING_DAYS,
            "min_sample": MIN_SAMPLE, "signal_accuracy": signal_acc, "analyst_accuracy": module_acc,
            "calibration": calib,
            "policy": "只報告不自動調權重；樣本 ≥ MIN_SAMPLE 且 95% 信賴區間不含 50% 才標 conclusive。"}


def _naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    i = pd.to_datetime(s.index)
    s.index = i.tz_localize(None) if i.tz is not None else i
    return s
