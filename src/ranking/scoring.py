"""Opportunity Score、狀態、進場區、輪動與部位大小。

Opportunity Score 回答「今天在這批股票裡，哪一支的 forward risk/reward 最好」，不是「哪家公司最好」：
  * 品質/成長只是其中幾項；短線過熱會直接扣分（overextension penalty），所以
    「品質 95、成長 98、催化 95、過熱 90」會落到 HOLD / BUY_ON_PULLBACK，而不是追高 BUY。
  * 缺資料的模組排除後重新加權（不當中性），並回報 coverage。
  * 市場狀態（regime）調整動能/估值/風險/產業動能的權重。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 權重依 2026-10-06 point-in-time 驗證（src/ranking/validate.py）決定：
#   * 價格面只有「6 個月風險調整動能排名」(momentum_rank) 在美/台股、樣本內外都穩定勝出；
#     多模組價格綜合分數反而較差 → 價格面以 momentum_rank 為核心。
#   * 產業動能：單獨看「看多 → 20 日跑贏大盤」57.7%（95% CI 56.1~59.2），但加進排名後組合報酬反而下降
#     （美股樣本內 40.2%→35.0%、樣本外 47.9%→40.4%）→ 權重 0，只當診斷資訊顯示。
#   * 相對強度 / 技術 / 低風險：消融測試未增加 alpha → 權重 0，只當診斷資訊顯示。
#   * 基本面、財報加速、估值、催化、AI 曝險：沒有 point-in-time 歷史無法回測 → 保留但權重有限，
#     並在輸出中標記「未經回測驗證」。
BASE_WEIGHTS = {
    "momentum_rank": 0.61,          # 已驗證（唯一在組合層級穩定增加 alpha 的價格因子）
    "earnings_acceleration": 0.10,  # 未驗證
    "valuation": 0.08,              # 未驗證
    "quality": 0.06,                # 未驗證
    "growth": 0.06,                 # 未驗證
    "catalyst": 0.05,               # 未驗證
    "ai_exposure": 0.04,            # 未驗證（專家先驗）
}
VALIDATED_COMPONENTS = ("momentum_rank",)
DIAGNOSTIC_ONLY = ("relative_strength", "technical", "low_risk", "industry_momentum")
# 過熱「不扣排名分數」（回測顯示扣分會降低報酬）；改由進場規則處理：
# 排名夠前但過熱 ≥ OVEREXTENSION_WAIT → BUY_ON_PULLBACK，等降溫再進（回測：報酬約持平，±1%）。
OVEREXTENSION_FREE = 55.0
OVEREXTENSION_PENALTY = 0.0
OVEREXTENSION_WAIT = 60.0
PRICE_ONLY_COMPONENTS = ("momentum_rank",)


def regime_weights(profile: Optional[Dict], exclude: tuple = ()) -> Dict[str, float]:
    p = profile or {}
    w = dict(BASE_WEIGHTS)
    w["momentum_rank"] *= p.get("momentum_w", 1.0)
    w["valuation"] *= p.get("valuation_w", 1.0)
    for k in exclude:
        w.pop(k, None)
    return w


def opportunity_score(components: Dict[str, Optional[float]], overextension: Optional[float],
                      weights: Dict[str, float]) -> Dict:
    used = {k: v for k, v in components.items() if k in weights and v is not None and np.isfinite(v)}
    if not used:
        return {"score": None, "coverage": 0.0, "penalty": 0.0, "raw": None}
    tot = sum(weights[k] for k in used)
    raw = sum(used[k] * weights[k] for k in used) / tot
    pen = max(0.0, (overextension or 0) - OVEREXTENSION_FREE) * OVEREXTENSION_PENALTY if overextension is not None else 0.0
    return {"score": round(max(0.0, min(100.0, raw - pen)), 1), "raw": round(raw, 1), "penalty": round(pen, 1),
            "coverage": round(tot / sum(weights.values()), 2), "components_used": sorted(used)}


def opportunity_panel(comp_panels: Dict[str, pd.DataFrame], overext: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """回測用的向量化版本（同一公式）。"""
    num, den = 0, 0
    for k, w in weights.items():
        if k not in comp_panels:
            continue
        p = comp_panels[k]
        num = num + p.fillna(0) * w
        den = den + p.notna().astype(float) * w
    raw = num / den.replace(0, np.nan)
    pen = ((overext - OVEREXTENSION_FREE).clip(lower=0) * OVEREXTENSION_PENALTY).fillna(0)
    return (raw - pen).clip(0, 100)


# ---------------------------------------------------------------------------
# 狀態
# ---------------------------------------------------------------------------
def decide_status(opp: Optional[float], overext: Optional[float], quality: Optional[float], risk: Optional[float],
                  valuation: Optional[float], relative_strength: Optional[float], held: bool,
                  damage: Optional[float] = None, oversold: bool = False, above_avoid: bool = False) -> Dict:
    flags = []
    if opp is None:
        return {"status": "WAIT", "flags": ["INSUFFICIENT_DATA"], "why": "資料不足，無法評分"}
    ox = overext or 0
    if oversold:
        flags.append("POTENTIAL_OVERSOLD")
    if quality is not None and quality >= 70 and ox >= 60:
        flags += ["BUY_QUALITY", "WAIT_ENTRY"]
    if held:
        if (opp < 30 and (relative_strength or 50) < 30) or (damage is not None and damage >= 60):
            return {"status": "SELL", "flags": flags, "why": "機會分數與相對強度同時轉弱，或基本面受損"}
        if ox >= 80 and (valuation is None or valuation < 50):
            return {"status": "PARTIAL_PROFIT", "flags": flags, "why": "短線過熱且估值不便宜，先分批獲利了結"}
        if quality is not None and quality >= 70 and opp >= 45:
            return {"status": "HOLD_CORE", "flags": flags, "why": "高品質核心部位，機會分數仍在中上"}
        if opp >= 30:
            return {"status": "HOLD", "flags": flags, "why": "機會分數中性，續抱觀察"}
        return {"status": "SELL", "flags": flags, "why": "機會分數偏低"}
    if opp >= 55 and above_avoid:
        if "WAIT_ENTRY" not in flags:
            flags.append("WAIT_ENTRY")
        return {"status": "BUY_ON_PULLBACK", "flags": flags, "why": "條件不錯，但現價已高於 avoid_above：等回到進場區再買"}
    if opp >= 65 and ox < OVEREXTENSION_WAIT:
        return {"status": "BUY_NOW", "flags": flags, "why": "機會分數高且未過熱"}
    if opp >= 55 and ox >= OVEREXTENSION_WAIT:
        if "WAIT_ENTRY" not in flags:
            flags.append("WAIT_ENTRY")
        return {"status": "BUY_ON_PULLBACK", "flags": flags,
                "why": "條件好但短線過熱：等降溫或回到進場區再買（回測：延後進場報酬不輸直接追）"}
    if opp >= 65:
        return {"status": "BUY_NOW", "flags": flags, "why": "機會分數高"}
    if opp >= 55:
        return {"status": "BUY_ON_PULLBACK", "flags": flags, "why": "機會分數中上：只在 Zone 1 以下分批，不追價"}
    if oversold and opp >= 50:
        return {"status": "BUY_ON_PULLBACK", "flags": flags, "why": "大跌但基本面受損低，分批於進場區承接"}
    return {"status": "WAIT", "flags": flags, "why": "機會分數不足，觀望"}


# ---------------------------------------------------------------------------
# 進場區（動態）
# ---------------------------------------------------------------------------
def historical_pullbacks(close: pd.Series, window: int = 60, years: int = 2) -> Dict[str, Optional[float]]:
    """過去 2 年在上升趨勢（站上 200 日線）中，自 60 日高點回檔的深度分布。"""
    c = close.dropna().tail(252 * years + 200)
    if len(c) < 260:
        return {"median": None, "p75": None}
    dd = c / c.rolling(window).max() - 1
    up = c > c.rolling(200).mean()
    troughs = []
    in_dd, low = False, 0.0
    for d, x in dd.items():
        if x < -0.03 and up.get(d, False):
            in_dd, low = True, min(low, x)
        elif in_dd and x > -0.01:
            troughs.append(low)
            in_dd, low = False, 0.0
    if len(troughs) < 3:
        return {"median": None, "p75": None}
    t = np.array(troughs)
    return {"median": float(np.median(t)), "p75": float(np.percentile(t, 25))}   # 25 百分位 = 較深的回檔


def entry_zones(ohlcv: pd.DataFrame, valuation_score: Optional[float] = None,
                days_to_earnings: Optional[int] = None) -> Dict:
    c, h, l = ohlcv["Close"].dropna(), ohlcv["High"].dropna(), ohlcv["Low"].dropna()
    if len(c) < 60:
        return {"zone_1": None, "zone_2": None, "zone_3": None, "avoid_above": None, "notes": ["資料不足"]}
    px = float(c.iloc[-1])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    ma20, ma50 = float(c.tail(20).mean()), float(c.tail(50).mean())
    ma200 = float(c.tail(200).mean()) if len(c) >= 200 else None
    low20, low60 = float(l.tail(20).min()), float(l.tail(60).min())
    hi60 = float(h.tail(60).max())
    lo252, hi252 = float(l.tail(252).min()), float(h.tail(252).max())
    pb = historical_pullbacks(c)
    cands = {"MA20": ma20, "MA50": ma50, "20 日低點": low20, "60 日低點": low60,
             "1 ATR": px - atr, "2 ATR": px - 2 * atr, "3 ATR": px - 3 * atr}
    if ma200:
        cands["MA200"] = ma200
    if pb["median"] is not None:
        cands["歷史中位回檔"] = hi60 * (1 + pb["median"])
        cands["歷史深回檔"] = hi60 * (1 + pb["p75"])
    shift = 0.5 * atr if (valuation_score is not None and valuation_score < 30) else 0.0
    below = sorted([(v - shift, k) for k, v in cands.items() if v < px * 0.995 and v > lo252 * 0.98], reverse=True)

    def pick(lo_bound: float, default: float, default_name: str):
        for v, k in below:
            if v >= lo_bound:
                return v, k
        return default, default_name

    z1, n1 = pick(px - 1.5 * atr, px - atr - shift, "1 ATR")
    rest = [(v, k) for v, k in below if v < z1 - 0.5 * atr]
    z2, n2 = next(((v, k) for v, k in rest if v >= px - 3.5 * atr), (z1 - atr, f"{n1}−1 ATR"))
    deeper = [(v, k) for v, k in below if v < z2 - 0.5 * atr]
    z3, n3 = deeper[0] if deeper else (max(z2 - 1.5 * atr, lo252), "52 週區間下緣")
    band = 0.25 * atr

    def zone(v, name):
        return {"low": round(v - band, 2), "high": round(v + band, 2), "basis": name,
                "pct_from_price": round((v / px - 1) * 100, 1)}

    avoid = min(ma20 + 2.5 * atr, max(px, hi252) * 1.02)
    notes = []
    if shift:
        notes.append("估值偏貴：各區下移 0.5 ATR")
    if days_to_earnings is not None and 0 <= days_to_earnings <= 10:
        notes.append(f"{days_to_earnings} 天後公布財報：Zone 1 只建小部位，財報後再決定")
    if px > avoid:
        notes.append("現價已高於 avoid_above，不追價")
    return {"zone_1": zone(z1, n1), "zone_2": zone(z2, n2), "zone_3": zone(z3, n3),
            "avoid_above": round(avoid, 2), "atr": round(atr, 2), "price": round(px, 2),
            "range_52w": [round(lo252, 2), round(hi252, 2)], "notes": notes}


# ---------------------------------------------------------------------------
# 預期報酬、輪動、部位
# ---------------------------------------------------------------------------
# 機會分數 → 預期 60 日超額報酬。預設為保守線性假設；P2 校準後由回測結果取代（calibration 檔）。
DEFAULT_CALIBRATION = {"slope_per_point": 0.0015, "center": 50.0, "source": "default linear prior (未校準)"}
ROUND_TRIP_COST = 0.004


def expected_excess_return(opp: Optional[float], calibration: Optional[Dict] = None) -> Optional[float]:
    if opp is None:
        return None
    cal = calibration or DEFAULT_CALIBRATION
    if "deciles" in cal:   # 由回測得到的分位數表
        edges, vals = cal["deciles"]["edges"], cal["deciles"]["mean_excess_60d"]
        i = int(np.searchsorted(edges, opp, side="right")) - 1
        return float(vals[max(0, min(i, len(vals) - 1))])
    return (opp - cal["center"]) * cal["slope_per_point"]


def rotation_suggestions(rows: List[Dict], calibration: Optional[Dict] = None, max_pairs: int = 5) -> List[Dict]:
    """SELL A → BUY B 的成對建議。rotation_score = 風險調整後的預期超額報酬差，再依估值/動能/催化調整。"""
    held = [r for r in rows if r.get("held")]
    cands = [r for r in rows if not r.get("held") and r["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")]
    out = []
    for a in held:
        ea = expected_excess_return(a.get("opportunity"), calibration)
        if ea is None:
            continue
        for b in cands:
            eb = expected_excess_return(b.get("opportunity"), calibration)
            if eb is None:
                continue
            va, vb = max(a.get("vol_63") or 0.4, 0.1), max(b.get("vol_63") or 0.4, 0.1)
            diff = eb / vb * 0.40 - ea / va * 0.40           # 換算成「40% 年化波動」的同風險基準
            adj = 0.0
            adj += 0.01 if (b["scores"].get("valuation") or 50) > (a["scores"].get("valuation") or 50) + 15 else 0.0
            adj += 0.01 if (b["scores"].get("catalyst") or 50) > (a["scores"].get("catalyst") or 50) + 15 else 0.0
            adj -= 0.01 if (b["scores"].get("overextension") or 0) >= 60 else 0.0
            score = diff + adj - ROUND_TRIP_COST
            if score <= 0.02:
                continue
            kind = "FULL_ROTATION" if (a["status"] == "SELL" or score > 0.08) else "PARTIAL_ROTATION"
            out.append({"sell": a["ticker"], "buy": b["ticker"], "rotation_score": round(score, 4), "type": kind,
                        "fraction_of_A": 1.0 if kind == "FULL_ROTATION" else 0.33,
                        "why": f"{b['ticker']} 風險調整後預期 60 日超額 {eb:+.1%} vs {a['ticker']} {ea:+.1%}"})
    out.sort(key=lambda x: x["rotation_score"], reverse=True)
    return out[:max_pairs]


def position_sizes(rows: List[Dict], profile: Dict, calibration: Optional[Dict] = None,
                   max_single: float = 0.12, max_theme: float = 0.35, max_high_beta_bucket: float = 0.50) -> Dict[str, float]:
    """目標權重 ∝ 預期超額報酬 / 預期風險 × 信心；套單檔、主題、因子（高 beta）上限與 regime 曝險。"""
    raw = {}
    for r in rows:
        if r["status"] not in ("BUY_NOW", "BUY_ON_PULLBACK", "HOLD", "HOLD_CORE"):
            continue
        e = expected_excess_return(r.get("opportunity"), calibration)
        if e is None or e <= 0:
            continue
        vol = max(r.get("vol_63") or 0.4, 0.15)
        downside = max(r.get("downside_risk") or vol, 0.10)
        raw[r["ticker"]] = e / (0.5 * vol + 0.5 * downside) * (r.get("coverage") or 0.5)
    if not raw:
        return {}
    tot = sum(raw.values())
    w = {k: v / tot * profile.get("exposure", 1.0) for k, v in raw.items()}
    hb_cap = profile.get("max_high_beta_weight", max_single)
    beta = {r["ticker"]: r.get("beta") or 1.0 for r in rows}
    theme = {r["ticker"]: r.get("theme") or "Other" for r in rows}
    for _ in range(5):   # 反覆套上限並把多出的權重按比例分給未觸頂者
        capped = {}
        for k, v in w.items():
            cap = min(max_single, hb_cap) if beta.get(k, 1) >= 1.5 else max_single
            capped[k] = min(v, cap)
        by_theme: Dict[str, float] = {}
        for k, v in capped.items():
            by_theme[theme[k]] = by_theme.get(theme[k], 0) + v
        for th, tv in by_theme.items():
            if tv > max_theme:
                for k in capped:
                    if theme[k] == th:
                        capped[k] *= max_theme / tv
        hb = sum(v for k, v in capped.items() if beta.get(k, 1) >= 1.5)
        if hb > max_high_beta_bucket:
            for k in capped:
                if beta.get(k, 1) >= 1.5:
                    capped[k] *= max_high_beta_bucket / hb
        w = capped
    return {k: round(v, 4) for k, v in sorted(w.items(), key=lambda x: -x[1]) if v >= 0.005}
