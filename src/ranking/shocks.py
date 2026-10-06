"""單日大跌診斷：大跌 ≠ 看空。先判斷原因，再估計基本面受損程度。

觸發：近 5 個交易日內單日跌幅 ≤ -5%（-8%、-12% 為更嚴重等級）。
原因（可複選，依同日對照組判斷，point-in-time 可回測）：
  macro            大盤同日 ≤ -2%，且個股跌幅大致可由 beta 解釋
  industry         同主題其他股票同日平均 ≤ -3%
  company_specific 扣掉 beta × 大盤與同主題後，仍有 ≤ -4% 的殘差
  technical        大跌前處於過熱（overextension ≥ 70）且無公司事件 → 獲利了結
  temporary / structural  由 fundamental_damage_score 判斷
fundamental_damage_score（0~100）：
  即時：近期負面事件類別（財報不如預期、下修財測、監管）、預估下修、surprise 為負
  回測：沒有 point-in-time 新聞 → 只用價格可得的部分（公司特有殘差、跌後是否持續破底）
若跌幅大但 damage 低 → POTENTIAL_OVERSOLD（不是 SELL）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SHOCK_LEVELS = (-0.05, -0.08, -0.12)
DAMAGING_CATEGORIES = {"earnings": 25, "guidance": 40, "regulatory": 30, "major_customer": 25,
                       "competitor_action": 15, "supply_increase": 20, "m_and_a": 10}


def detect_shocks(ret_1: pd.Series, spy_ret_1: pd.Series, theme_ret_1: Optional[pd.Series], beta: Optional[float],
                  overextension_before: Optional[float], lookback: int = 5) -> List[Dict]:
    """ret_1 等為同一股票的日報酬序列（index 對齊）。回傳近 lookback 日的大跌事件。"""
    out = []
    r = ret_1.dropna().tail(lookback)
    for d, x in r.items():
        if x > SHOCK_LEVELS[0]:
            continue
        m = float(spy_ret_1.get(d, np.nan)) if spy_ret_1 is not None else np.nan
        p = float(theme_ret_1.get(d, np.nan)) if theme_ret_1 is not None else np.nan
        b = beta if (beta is not None and np.isfinite(beta)) else 1.0
        explained = (b * m if np.isfinite(m) else 0.0)
        residual = x - explained - (0.5 * (p - explained) if np.isfinite(p) else 0.0)
        causes = []
        if np.isfinite(m) and m <= -0.02:
            causes.append("macro")
        if np.isfinite(p) and p <= -0.03:
            causes.append("industry")
        if residual <= -0.04:
            causes.append("company_specific")
        if overextension_before is not None and overextension_before >= 70 and "company_specific" not in causes:
            causes.append("technical")
        level = sum(1 for lv in SHOCK_LEVELS if x <= lv)
        out.append({"date": pd.Timestamp(d).date().isoformat(), "return": round(float(x), 4), "severity": level,
                    "market_return": None if not np.isfinite(m) else round(m, 4),
                    "theme_return": None if not np.isfinite(p) else round(p, 4),
                    "residual": round(float(residual), 4), "causes": causes or ["unexplained"]})
    return out


def fundamental_damage(shock: Dict, catalyst_events: Optional[List[Dict]], eps_revision_90d: Optional[float],
                       latest_surprise: Optional[float], recovered_pct: Optional[float]) -> Dict:
    """估計基本面受損程度 0~100。catalyst_events 為 None 代表無 point-in-time 新聞（回測），只用價格證據。"""
    score, why = 0.0, []
    if "company_specific" in shock["causes"]:
        score += 25
        why.append("公司特有跌幅（扣除大盤與同業後仍大跌）")
    if catalyst_events:
        for e in catalyst_events:
            if e.get("direction", 0) < 0 and e.get("age_days", 99) <= 7:
                w = DAMAGING_CATEGORIES.get(e.get("category"), 0)
                if w:
                    score += w
                    why.append(f"負面事件：{e['category']}")
    if eps_revision_90d is not None and eps_revision_90d < -0.03:
        score += 20
        why.append(f"EPS 預估 90 天下修 {eps_revision_90d:.0%}")
    if latest_surprise is not None and latest_surprise < -5:
        score += 15
        why.append(f"最近一季 EPS 不如預期 {latest_surprise:.0f}%")
    if recovered_pct is not None and recovered_pct < -0.03:
        score += 10
        why.append("跌後持續破底")
    if set(shock["causes"]) <= {"macro", "industry", "technical"}:
        score -= 15
        why.append("主因為大盤/產業/技術面，非公司本身")
    score = float(max(0.0, min(100.0, score)))
    structural = score >= 50
    return {"fundamental_damage_score": round(score), "structural": structural,
            "nature": "structural" if structural else "temporary", "evidence": why,
            "point_in_time_news": catalyst_events is not None}


def shock_verdict(shock: Dict, damage: Dict) -> Optional[str]:
    if shock["severity"] >= 2 and damage["fundamental_damage_score"] < 40:
        return "POTENTIAL_OVERSOLD"
    if shock["severity"] >= 1 and damage["fundamental_damage_score"] < 25:
        return "POTENTIAL_OVERSOLD"
    if damage["fundamental_damage_score"] >= 60:
        return "THESIS_AT_RISK"
    return None
