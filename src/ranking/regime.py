"""市場狀態：BULL / NEUTRAL / RISK_OFF / RISK_ON_ROTATION（point-in-time，可回測）。

輸入（截至 asof 的收盤）：SPY、QQQ、^VIX、^TNX（10 年期殖利率）、美元指數 DX-Y.NYB、
SMH（半導體）、XLK（科技）、XLU（公用/電力）、XLE（能源）。

規則（每一條都有經濟意義，門檻為常見慣例值而非擬合）：
  RISK_OFF         SPY 跌破 200 日線，或 VIX ≥ 28，或 SPY 20 日 ≤ -7%
  BULL             SPY 在 200 日線上、VIX < 20、QQQ 60 日相對 SPY ≥ 0（成長股領漲）
  RISK_ON_ROTATION SPY 在 200 日線上，但 XLU 或 XLE 60 日相對 QQQ 領先 ≥ 3%（資金從科技轉向電力/能源）
  NEUTRAL          其餘

狀態影響（REGIME_PROFILES）：部位乘數、動能/估值權重倍數、單一高 beta 部位上限。
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

TICKERS = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "TNX": "^TNX", "USD": "DX-Y.NYB",
           "SMH": "SMH", "XLK": "XLK", "XLU": "XLU", "XLE": "XLE"}

# exposure 一律 1.0：point-in-time 回測顯示依狀態降低總曝險會降低 Sharpe（美股 1.02→0.84）。
# 狀態只影響權重倍數與「高 beta 單檔上限」等風險限制。
REGIME_PROFILES = {
    "BULL":             {"exposure": 1.00, "momentum_w": 1.25, "valuation_w": 0.75, "risk_w": 0.75, "max_high_beta_weight": 0.12},
    "NEUTRAL":          {"exposure": 1.00, "momentum_w": 1.00, "valuation_w": 1.00, "risk_w": 1.00, "max_high_beta_weight": 0.10},
    "RISK_ON_ROTATION": {"exposure": 1.00, "momentum_w": 0.90, "valuation_w": 1.10, "risk_w": 1.00, "max_high_beta_weight": 0.08,
                         "industry_w": 1.30},
    "RISK_OFF":         {"exposure": 1.00, "momentum_w": 0.70, "valuation_w": 1.35, "risk_w": 1.50, "max_high_beta_weight": 0.05},
}


def _ret(s: pd.Series, n: int) -> Optional[float]:
    s = s.dropna()
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1) if len(s) > n else None


def classify_regime(closes: Dict[str, pd.Series], asof: Optional[pd.Timestamp] = None) -> Dict:
    c = {}
    for k, s in closes.items():
        if s is None:
            continue
        s = s.copy()
        i = pd.to_datetime(s.index)
        s.index = i.tz_localize(None) if i.tz is not None else i
        c[k] = s.loc[:asof] if asof is not None else s
    spy = c.get("SPY")
    if spy is None or len(spy.dropna()) < 200:
        return {"regime": "insufficient_data", "profile": REGIME_PROFILES["NEUTRAL"], "metrics": {}}
    spy = spy.dropna()
    ma200 = float(spy.tail(200).mean())
    m = {
        "spy_vs_ma200": float(spy.iloc[-1] / ma200 - 1),
        "spy_20d": _ret(spy, 20),
        "vix": float(c["VIX"].dropna().iloc[-1]) if "VIX" in c and len(c["VIX"].dropna()) else None,
        "tnx_20d_change_bp": (float(c["TNX"].dropna().iloc[-1] - c["TNX"].dropna().iloc[-21]) * 10) if "TNX" in c and len(c["TNX"].dropna()) > 21 else None,
        "usd_20d": _ret(c["USD"], 20) if "USD" in c else None,
    }
    spy60 = _ret(spy, 60)
    for k in ("QQQ", "SMH", "XLK", "XLU", "XLE"):
        if k in c:
            r = _ret(c[k], 60)
            m[f"{k.lower()}_rel_spy_60d"] = (r - spy60) if (r is not None and spy60 is not None) else None
    qqq_rel = m.get("qqq_rel_spy_60d")
    rot = max([(m.get(f"{k}_rel_spy_60d") or -9) - (qqq_rel or 0) for k in ("xlu", "xle")])
    if m["spy_vs_ma200"] < 0 or (m["vix"] or 0) >= 28 or (m["spy_20d"] or 0) <= -0.07:
        regime = "RISK_OFF"
    elif rot >= 0.03:
        regime = "RISK_ON_ROTATION"
    elif (m["vix"] or 99) < 20 and (qqq_rel or -1) >= 0:
        regime = "BULL"
    else:
        regime = "NEUTRAL"
    m["rotation_spread"] = rot if rot > -9 else None
    return {"regime": regime, "profile": REGIME_PROFILES[regime], "metrics": m}


def fetch_regime_inputs(period: str = "2y") -> Dict[str, pd.Series]:
    from src.strategy.momentum import download_closes
    pm = download_closes(list(TICKERS.values()), period=period)
    return {k: pm[v]["Close"] for k, v in TICKERS.items() if v in pm}
