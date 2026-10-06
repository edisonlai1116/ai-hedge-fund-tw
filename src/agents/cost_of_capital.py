"""資金成本（WACC）：信用利差模型、point-in-time beta 與市場無風險利率。

2026-10-06 修正舊版 valuation.calculate_wacc 的錯誤：
  * 舊式 cost_of_debt = rf + 10 / interest_coverage → coverage=5 時利差 200%、coverage=2 時 500%，
    大量公司的 WACC 被頂到 20% 上限，DCF 系統性低估。
  * 改用「利息保障倍數 → 合成信用評等 → 信用利差」（Damodaran 方法），利差限制在 1%~10%。
  * beta 不再固定 1.0：以 point-in-time 週報酬對 SPY 回歸（只用 end_date 以前的價格）。
  * 無風險利率取 end_date 當時的美國 10 年期公債殖利率（^TNX），取不到才用設定值。
  * 市場風險溢酬可由環境變數 MARKET_RISK_PREMIUM 設定（預設 5%）。
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

MIN_CREDIT_SPREAD = 0.01
MAX_CREDIT_SPREAD = 0.10
DEFAULT_CREDIT_SPREAD = 0.025      # 無 interest coverage 資料：約 BBB-
DEFAULT_RISK_FREE = float(os.environ.get("RISK_FREE_RATE", "0.043"))
MARKET_RISK_PREMIUM = float(os.environ.get("MARKET_RISK_PREMIUM", "0.05"))
TAX_RATE = float(os.environ.get("CORPORATE_TAX_RATE", "0.21"))
BETA_BOUNDS = (0.3, 3.0)
WACC_SANITY_BOUNDS = (0.04, 0.25)  # 只擋明顯錯誤資料；正常公司不應碰到

# 合成評等利差表（大型企業版，依利息保障倍數下限由高到低）。
_COVERAGE_SPREADS = [
    (8.5, 0.0070), (6.5, 0.0090), (5.5, 0.0110), (4.25, 0.0125), (3.0, 0.0140),
    (2.5, 0.0180), (2.25, 0.0230), (2.0, 0.0280), (1.75, 0.0350), (1.5, 0.0420),
    (1.25, 0.0500), (0.8, 0.0750), (0.65, 0.0900), (float("-inf"), 0.1000),
]


def credit_spread_by_interest_coverage(interest_coverage: float | None) -> float:
    if interest_coverage is None or not np.isfinite(interest_coverage):
        return DEFAULT_CREDIT_SPREAD
    for floor, spread in _COVERAGE_SPREADS:
        if interest_coverage >= floor:
            return min(max(spread, MIN_CREDIT_SPREAD), MAX_CREDIT_SPREAD)
    return MAX_CREDIT_SPREAD


def cost_of_debt(risk_free_rate: float, interest_coverage: float | None) -> float:
    return risk_free_rate + credit_spread_by_interest_coverage(interest_coverage)


@lru_cache(maxsize=256)
def risk_free_rate_asof(end_date: str) -> tuple[float, str]:
    """end_date 當時（含以前最後一筆）的 10 年期美債殖利率。回傳 (rate, source)。"""
    try:
        import yfinance as yf
        end = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
        hist = yf.Ticker("^TNX").history(start=(end - timedelta(days=14)).isoformat(),
                                        end=(end + timedelta(days=1)).isoformat())
        closes = hist["Close"].dropna()
        closes = closes[closes.index.date <= end]
        if len(closes):
            return float(closes.iloc[-1]) / 100.0, f"^TNX {closes.index[-1].date()}"
    except Exception:
        pass
    return DEFAULT_RISK_FREE, "config RISK_FREE_RATE"


def estimate_beta(stock_close: pd.Series, market_close: pd.Series, min_weeks: int = 52) -> float | None:
    """週報酬 OLS beta（只用呼叫端給的 point-in-time 價格）。資料不足回 None。"""
    s = pd.Series(stock_close).dropna()
    m = pd.Series(market_close).dropna()
    if s.empty or m.empty:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
    m.index = pd.to_datetime(m.index).tz_localize(None) if getattr(m.index, "tz", None) else pd.to_datetime(m.index)
    df = pd.concat([s.resample("W-FRI").last(), m.resample("W-FRI").last()], axis=1).dropna().pct_change().dropna()
    if len(df) < min_weeks:
        return None
    var = float(df.iloc[:, 1].var())
    if var <= 0:
        return None
    beta = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var)
    return float(min(max(beta, BETA_BOUNDS[0]), BETA_BOUNDS[1]))


def beta_asof(ticker: str, end_date: str, api_key: str | None = None) -> tuple[float | None, str]:
    """用 end_date 以前 2 年價格估 beta（point-in-time）。失敗回 (None, reason)。"""
    try:
        from src.tools.api import get_price_data
        start = (datetime.strptime(end_date[:10], "%Y-%m-%d") - timedelta(days=740)).strftime("%Y-%m-%d")
        s = get_price_data(ticker, start, end_date, api_key=api_key)["close"]
        m = get_price_data("SPY", start, end_date, api_key=api_key)["close"]
        b = estimate_beta(s, m)
        return (b, "2y weekly vs SPY") if b is not None else (None, "insufficient price history")
    except Exception as exc:
        return None, f"beta unavailable: {type(exc).__name__}"


def calculate_wacc(
    market_cap: float,
    total_debt: float | None,
    cash: float | None,
    interest_coverage: float | None,
    beta: float | None,
    risk_free_rate: float,
    market_risk_premium: float = MARKET_RISK_PREMIUM,
    tax_rate: float = TAX_RATE,
) -> dict:
    """回傳 WACC 與各組成（方便在 reasoning 中呈現、也方便測試）。beta 缺值時以 1.0 並標記。"""
    beta_used = beta if beta is not None else 1.0
    coe = risk_free_rate + beta_used * market_risk_premium
    spread = credit_spread_by_interest_coverage(interest_coverage)
    cod = risk_free_rate + spread
    net_debt = max((total_debt or 0) - (cash or 0), 0)
    total_value = (market_cap or 0) + net_debt
    if total_value > 0:
        we, wd = market_cap / total_value, net_debt / total_value
        raw = we * coe + wd * cod * (1 - tax_rate)
    else:
        we, wd, raw = 1.0, 0.0, coe
    wacc = min(max(raw, WACC_SANITY_BOUNDS[0]), WACC_SANITY_BOUNDS[1])
    return {
        "wacc": wacc,
        "wacc_unbounded": raw,
        "bounded": wacc != raw,
        "cost_of_equity": coe,
        "cost_of_debt": cod,
        "credit_spread": spread,
        "beta": beta_used,
        "beta_is_default": beta is None,
        "risk_free_rate": risk_free_rate,
        "market_risk_premium": market_risk_premium,
        "weight_equity": we,
        "weight_debt": wd,
    }
