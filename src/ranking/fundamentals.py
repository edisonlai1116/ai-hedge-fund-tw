"""基本面評分（即時；yfinance 只提供「現在」的財報與預估 → 不進回測，回測中標記為 not_point_in_time）。

  quality_score              毛利、營益率、ROE、FCF 率、槓桿
  growth_score               營收 / EPS 成長（含前瞻預估）
  earnings_acceleration_score  重點是「成長是否正在加速」：營收/EPS 年增率的變化、利潤率擴張、
                             最近一季 surprise、預估上修（guidance/revision 代理）
  valuation_score            依 A~F 類別用不同框架；高分 = 便宜

所有分數 0~100；資料不足的子項排除後重新加權，全部不足 → None（不當中性）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ranking.themes import classify_valuation


def _ramp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    v = (x - lo) / (hi - lo) * 100
    return float(min(100.0, max(0.0, v)))


def _blend(parts: Dict[str, Tuple[Optional[float], float]]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    used = {k: (v, w) for k, (v, w) in parts.items() if v is not None}
    if not used:
        return None, {k: None for k in parts}
    tot = sum(w for _, w in used.values())
    return round(sum(v * w for v, w in used.values()) / tot, 1), {k: (round(v, 1) if v is not None else None) for k, (v, _) in parts.items()}


def _row(df: Optional[pd.DataFrame], names: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s = s.sort_index(ascending=False)       # 最新在前
            return s
    return None


def _yoy(s: Optional[pd.Series], i: int = 0) -> Optional[float]:
    """第 i 新一季對去年同季的年增率（季資料需 >= i+5 筆）。"""
    if s is None or len(s) < i + 5:
        return None
    cur, prev = s.iloc[i], s.iloc[i + 4]
    if pd.isna(cur) or pd.isna(prev) or prev <= 0:
        return None
    return float(cur / prev - 1)


def _margin(num: Optional[pd.Series], rev: Optional[pd.Series], i: int) -> Optional[float]:
    if num is None or rev is None or len(num) <= i or len(rev) <= i:
        return None
    n, r = num.iloc[i], rev.iloc[i]
    if pd.isna(n) or pd.isna(r) or r <= 0:
        return None
    return float(n / r)


class FundamentalSnapshot:
    """抓一次 yfinance 資料後計算所有基本面分數。"""

    def __init__(self, symbol: str):
        import yfinance as yf
        self.symbol = symbol
        t = yf.Ticker(symbol)
        self.info: Dict[str, Any] = _safe(lambda: t.info, {}) or {}
        self.q_inc = _safe(lambda: t.quarterly_income_stmt, None)
        self.q_cf = _safe(lambda: t.quarterly_cashflow, None)
        self.earn_dates = _safe(lambda: t.earnings_dates, None)
        self.eps_trend = _safe(lambda: t.eps_trend, None)
        self.eps_rev = _safe(lambda: t.eps_revisions, None)
        self.rev_est = _safe(lambda: t.revenue_estimate, None)
        self.earn_est = _safe(lambda: t.earnings_estimate, None)
        self.valuation_class = classify_valuation(symbol, self.info)

    # ---------------- quality ----------------
    def quality(self) -> Dict[str, Any]:
        i = self.info
        fcf_margin = None
        if i.get("freeCashflow") and i.get("totalRevenue"):
            fcf_margin = i["freeCashflow"] / i["totalRevenue"]
        nd_ebitda = None
        if i.get("ebitda") and i["ebitda"] > 0:
            nd_ebitda = ((i.get("totalDebt") or 0) - (i.get("totalCash") or 0)) / i["ebitda"]
        score, parts = _blend({
            "gross_margin": (_ramp(i.get("grossMargins"), 0.15, 0.70), 0.20),
            "operating_margin": (_ramp(i.get("operatingMargins"), 0.0, 0.40), 0.25),
            "roe": (_ramp(i.get("returnOnEquity"), 0.0, 0.35), 0.20),
            "fcf_margin": (_ramp(fcf_margin, -0.05, 0.30), 0.20),
            "leverage": (_ramp(nd_ebitda, 4.0, 0.0) if nd_ebitda is not None else None, 0.15),
        })
        return {"score": score, "parts": parts, "metrics": {"fcf_margin": fcf_margin, "net_debt_to_ebitda": nd_ebitda}}

    # ---------------- growth ----------------
    def growth(self) -> Dict[str, Any]:
        i = self.info
        fwd_rev = _est_growth(self.rev_est, "+1y")
        fwd_eps = _est_growth(self.earn_est, "+1y")
        score, parts = _blend({
            "revenue_growth": (_ramp(i.get("revenueGrowth"), -0.05, 0.60), 0.30),
            "eps_growth": (_ramp(i.get("earningsGrowth"), -0.10, 0.80), 0.20),
            "fwd_revenue_growth": (_ramp(fwd_rev, 0.0, 0.50), 0.30),
            "fwd_eps_growth": (_ramp(fwd_eps, 0.0, 0.60), 0.20),
        })
        return {"score": score, "parts": parts,
                "metrics": {"revenue_growth": i.get("revenueGrowth"), "eps_growth": i.get("earningsGrowth"),
                            "fwd_revenue_growth": fwd_rev, "fwd_eps_growth": fwd_eps}}

    # ---------------- earnings acceleration ----------------
    def earnings_acceleration(self) -> Dict[str, Any]:
        rev = _row(self.q_inc, ["Total Revenue", "Operating Revenue"])
        eps = _row(self.q_inc, ["Diluted EPS", "Basic EPS"])
        op = _row(self.q_inc, ["Operating Income", "Total Operating Income As Reported"])
        gp = _row(self.q_inc, ["Gross Profit"])
        fcf = _row(self.q_cf, ["Free Cash Flow"])
        rev_yoy0, rev_yoy1 = _yoy(rev, 0), _yoy(rev, 1)
        eps_yoy0, eps_yoy1 = _yoy(eps, 0), _yoy(eps, 1)
        fcf_yoy = _yoy(fcf, 0)
        # 季資料不足 5 季時，用預估：本季（0q）預估成長 vs 下季（+1q）預估成長
        est_rev_now, est_rev_next = _est_growth(self.rev_est, "0q"), _est_growth(self.rev_est, "+1q")
        rev_accel = (rev_yoy0 - rev_yoy1) if (rev_yoy0 is not None and rev_yoy1 is not None) else (
            (est_rev_next - est_rev_now) if (est_rev_next is not None and est_rev_now is not None) else None)
        eps_accel = (eps_yoy0 - eps_yoy1) if (eps_yoy0 is not None and eps_yoy1 is not None) else None
        om_exp = _diff(_margin(op, rev, 0), _margin(op, rev, 4))
        gm_exp = _diff(_margin(gp, rev, 0), _margin(gp, rev, 4))
        surprise = _latest_surprise(self.earn_dates)
        revision = _revision_pct(self.eps_trend)
        rev_breadth = _revision_breadth(self.eps_rev)
        score, parts = _blend({
            "revenue_growth": (_ramp(rev_yoy0, -0.05, 0.60), 0.10),
            "revenue_acceleration": (_ramp(rev_accel, -0.15, 0.15), 0.20),
            "eps_growth": (_ramp(eps_yoy0, -0.10, 0.80), 0.08),
            "eps_acceleration": (_ramp(eps_accel, -0.30, 0.30), 0.12),
            "fcf_growth": (_ramp(fcf_yoy, -0.20, 0.60), 0.05),
            "operating_margin_expansion": (_ramp(om_exp, -0.05, 0.05), 0.12),
            "gross_margin_expansion": (_ramp(gm_exp, -0.04, 0.04), 0.08),
            "latest_surprise": (_ramp(surprise, -10.0, 15.0), 0.10),
            "guidance_revision": (_ramp(revision, -0.08, 0.08), 0.10),
            "revision_breadth": (_ramp(rev_breadth, -0.6, 0.6), 0.05),
        })
        accelerating = None
        if rev_accel is not None:
            accelerating = rev_accel > 0.02 and (revision is None or revision >= 0)
        return {
            "score": score, "parts": parts, "accelerating": accelerating,
            "metrics": {
                "revenue_yoy_latest": rev_yoy0, "revenue_yoy_prior": rev_yoy1, "revenue_acceleration": rev_accel,
                "eps_yoy_latest": eps_yoy0, "eps_yoy_prior": eps_yoy1, "eps_acceleration": eps_accel,
                "fcf_yoy": fcf_yoy, "operating_margin_expansion": om_exp, "gross_margin_expansion": gm_exp,
                "latest_surprise_pct": surprise, "eps_estimate_revision_90d": revision,
                "revision_breadth": rev_breadth, "quarters_available": int(len(rev)) if rev is not None else 0,
            },
        }

    # ---------------- valuation（依類別） ----------------
    def valuation(self, wacc: Optional[float] = None) -> Dict[str, Any]:
        i, cls = self.info, self.valuation_class
        mcap, ev = i.get("marketCap"), i.get("enterpriseValue")
        fcf_yield = (i["freeCashflow"] / mcap) if (i.get("freeCashflow") and mcap) else None
        fpe, ev_s, ev_e = i.get("forwardPE"), i.get("enterpriseToRevenue"), i.get("enterpriseToEbitda")
        g_rev = _est_growth(self.rev_est, "+1y") or i.get("revenueGrowth")
        g_eps = _est_growth(self.earn_est, "+1y") or i.get("earningsGrowth")
        peg = (fpe / (g_eps * 100)) if (fpe and fpe > 0 and g_eps and g_eps > 0.02) else None
        ev_s_g = (ev_s / (g_rev * 100)) if (ev_s and g_rev and g_rev > 0.02) else None
        notes: List[str] = []
        if cls == "A":
            parts = {
                "forward_pe": (_ramp(fpe, 80, 15) if fpe and fpe > 0 else None, 0.20),
                "peg": (_ramp(peg, 3.0, 0.7), 0.25),
                "ev_sales_per_growth": (_ramp(ev_s_g, 0.6, 0.08), 0.20),
                "ev_ebitda": (_ramp(ev_e, 80, 15) if ev_e and ev_e > 0 else None, 0.15),
                "fcf_yield": (_ramp(fcf_yield, -0.01, 0.05), 0.20),
            }
            notes.append("高成長/AI：用前瞻本益比、PEG、成長調整 EV/Sales、FCF yield，不用固定 DCF。")
        elif cls == "B":
            parts = {
                "forward_pe": (_ramp(fpe, 45, 12) if fpe and fpe > 0 else None, 0.30),
                "peg": (_ramp(peg, 2.5, 0.8), 0.25),
                "ev_ebitda": (_ramp(ev_e, 40, 10) if ev_e and ev_e > 0 else None, 0.20),
                "fcf_yield": (_ramp(fcf_yield, 0.0, 0.06), 0.25),
            }
        elif cls == "C":
            dcf_gap = _simple_dcf_gap(i, g_rev, wacc)
            parts = {
                "dcf_gap": (_ramp(dcf_gap, -0.40, 0.40), 0.30),
                "fcf_yield": (_ramp(fcf_yield, 0.01, 0.08), 0.30),
                "ev_ebitda": (_ramp(ev_e, 25, 8) if ev_e and ev_e > 0 else None, 0.25),
                "forward_pe": (_ramp(fpe, 30, 10) if fpe and fpe > 0 else None, 0.15),
            }
            notes.append("成熟型：DCF（兩階段、成長隨時間收斂）+ FCF yield + EV/EBITDA。")
        elif cls == "D":
            peak_ratio = self._peak_margin_ratio()
            norm_pe = (fpe * peak_ratio) if (fpe and fpe > 0 and peak_ratio) else fpe
            parts = {
                "normalized_pe": (_ramp(norm_pe, 35, 8) if norm_pe and norm_pe > 0 else None, 0.30),
                "ev_ebitda": (_ramp(ev_e, 25, 6) if ev_e and ev_e > 0 else None, 0.30),
                "normalized_fcf_yield": (_ramp(fcf_yield / peak_ratio if (fcf_yield and peak_ratio) else fcf_yield, 0.0, 0.08), 0.25),
                "peak_margin_penalty": (_ramp(peak_ratio, 1.6, 1.0) if peak_ratio else None, 0.15),
            }
            notes.append("循環股：用正常化（mid-cycle）利潤率估值，避免把景氣高峰獲利當永久。")
        elif cls == "F":
            nd_ebitda = (((i.get("totalDebt") or 0) - (i.get("totalCash") or 0)) / i["ebitda"]) if i.get("ebitda") else None
            parts = {
                "ev_ebitda": (_ramp(ev_e, 25, 8) if ev_e and ev_e > 0 else None, 0.40),
                "fcf_yield": (_ramp(fcf_yield, 0.0, 0.07), 0.30),
                "debt_to_ebitda": (_ramp(nd_ebitda, 6.0, 1.5) if nd_ebitda is not None else None, 0.30),
            }
            notes.append("電力/公用事業：EV/EBITDA、FCF、負債。裝置容量、PPA 合約、利率敏感度需另行資料，"
                         "免費資料源沒有，標記為 not_available（不捏造）。")
        else:  # E financial
            pb = i.get("priceToBook")
            parts = {
                "price_to_book": (_ramp(pb, 3.0, 0.8) if pb else None, 0.5),
                "forward_pe": (_ramp(fpe, 20, 7) if fpe and fpe > 0 else None, 0.5),
            }
        score, p = _blend(parts)
        return {
            "score": score, "parts": p, "valuation_class": cls, "notes": notes,
            "metrics": {"forward_pe": fpe, "peg": peg, "ev_sales": ev_s, "ev_sales_per_growth": ev_s_g,
                        "ev_ebitda": ev_e, "fcf_yield": fcf_yield, "fwd_revenue_growth": g_rev, "fwd_eps_growth": g_eps},
            "not_available": ["power_capacity", "ppa_contracts", "interest_sensitivity"] if cls == "F" else [],
        }

    def _peak_margin_ratio(self) -> Optional[float]:
        """目前營益率 / 可得期間平均營益率（> 1 代表處於景氣高點）。"""
        rev = _row(self.q_inc, ["Total Revenue", "Operating Revenue"])
        op = _row(self.q_inc, ["Operating Income", "Total Operating Income As Reported"])
        if rev is None or op is None or len(rev) < 4:
            return None
        m = (op / rev).dropna()
        if len(m) < 4 or m.mean() <= 0:
            return None
        return float(max(0.5, min(3.0, m.iloc[0] / m.mean())))

    def next_earnings_date(self) -> Optional[pd.Timestamp]:
        ed = self.earn_dates
        if ed is None or ed.empty:
            return None
        now = pd.Timestamp.now(tz=ed.index.tz)
        future = [d for d in ed.index if d >= now]
        return min(future) if future else None


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return (a - b) if (a is not None and b is not None) else None


def _est_growth(df: Optional[pd.DataFrame], period: str) -> Optional[float]:
    try:
        v = df.loc[period, "growth"]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _latest_surprise(ed: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        rep = ed.dropna(subset=["Reported EPS"])
        return float(rep["Surprise(%)"].iloc[0]) if len(rep) else None
    except Exception:
        return None


def _revision_pct(trend: Optional[pd.DataFrame]) -> Optional[float]:
    """本年與明年 EPS 預估「現在 vs 90 天前」的平均變化率：前瞻指引/分析師上修的代理。"""
    try:
        vals = []
        for p in ("0y", "+1y"):
            cur, old = trend.loc[p, "current"], trend.loc[p, "90daysAgo"]
            if pd.notna(cur) and pd.notna(old) and abs(old) > 1e-9:
                vals.append(cur / old - 1)
        return float(np.mean(vals)) if vals else None
    except Exception:
        return None


def _revision_breadth(rev: Optional[pd.DataFrame]) -> Optional[float]:
    """近 30 天上修家數 − 下修家數，除以總數（-1~+1）。"""
    try:
        up = rev["upLast30days"].fillna(0).sum()
        dn = rev["downLast30days"].fillna(0).sum()
        return float((up - dn) / (up + dn)) if (up + dn) > 0 else None
    except Exception:
        return None


def _simple_dcf_gap(info: Dict[str, Any], growth: Optional[float], wacc: Optional[float]) -> Optional[float]:
    """兩階段 DCF（成熟型專用）：前 5 年成長由 growth 線性收斂到 3%，終值成長 2.5%。
    growth 不設 10% 硬上限（成長本來就會收斂），回傳「內在價值 / 市值 − 1」。"""
    fcf, mcap = info.get("freeCashflow"), info.get("marketCap")
    if not fcf or fcf <= 0 or not mcap:
        return None
    r = wacc or 0.09
    g0 = growth if growth is not None else 0.04
    g0 = max(-0.10, min(g0, 0.60))
    pv, cf = 0.0, fcf
    for yr in range(1, 11):
        g = g0 + (0.03 - g0) * min(yr - 1, 9) / 9
        cf *= 1 + g
        pv += cf / (1 + r) ** yr
    tv = cf * 1.025 / max(r - 0.025, 0.02)
    pv += tv / (1 + r) ** 10
    net_cash = (info.get("totalCash") or 0) - (info.get("totalDebt") or 0)
    return float((pv + net_cash) / mcap - 1)
