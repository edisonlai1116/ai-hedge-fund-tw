"""Sharpe 動能輪動策略的離線單元測試（不連網）。"""
import numpy as np
import pandas as pd

from src.strategy import momentum as m


def _report(rows_us):
    rows = []
    for i, (sym, score) in enumerate(rows_us, 1):
        rows.append({"symbol": sym, "name": "", "rank": i, "zone": m.zone_of(i, "us"), "score": score,
                     "close": 100.0, "ret_6m_pct": 10.0})
    return {"generated_at": "2026-10-06T00:00:00+08:00",
            "strategy": {"next_rebalance": "2026-11-02", "in_rebalance_window": False},
            "markets": {"us": {"universe_size": len(rows), "rows": rows}, "tw": {"universe_size": 0, "rows": []}}}


def test_sharpe_momentum_prefers_smooth_uptrend():
    idx = pd.bdate_range("2025-01-01", periods=200)
    rng = np.random.default_rng(0)
    smooth = pd.Series(100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.005, 200)), index=idx)
    choppy = pd.Series(100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.03, 200)), index=idx)
    assert m.sharpe_momentum(smooth) > m.sharpe_momentum(choppy)
    assert m.sharpe_momentum(smooth.iloc[:50]) is None


def test_zones_use_market_buffers():
    assert m.zone_of(10, "us") == "buy"
    assert m.zone_of(11, "us") == "hold"
    assert m.zone_of(50, "us") == "hold"
    assert m.zone_of(51, "us") == "out"
    assert m.zone_of(20, "tw") == "buy"
    assert m.zone_of(31, "tw") == "out"


def test_evaluate_holdings_actions():
    syms = [(f"S{i:03d}", 5 - i * 0.01) for i in range(150)]
    rep = _report(syms)
    holdings = [
        {"ticker": "S000", "cost": 50, "shares": 1},     # 第 1 名、部位很小 → 加碼
        {"ticker": "S030", "cost": 50, "shares": 10},    # 第 31 名 → 續抱
        {"ticker": "S120", "cost": 50, "shares": 10},    # 第 121 名（> 50）→ 賣出換股
        {"ticker": "VOO", "cost": 400, "shares": 10},    # ETF → 核心（也讓單檔佔比 < 20%）
    ]
    ev = m.evaluate_holdings(holdings, rep, fx_usd_twd=30.0,
                             extra_prices={"VOO": pd.DataFrame({"Close": [500.0] * 10})})
    acts = {h["symbol"]: h["action"] for h in ev["holdings"]}
    assert acts == {"S000": "加碼", "S030": "續抱", "S120": "賣出換股", "VOO": "核心 ETF"}
    assert [b["symbol"] for b in ev["new_buys"]["us"]][:2] == ["S001", "S002"]


def test_rebalance_window_first_three_weekdays():
    from datetime import date
    assert m.is_rebalance_window(date(2026, 11, 2))      # 週一，第 1 個平日
    assert m.is_rebalance_window(date(2026, 11, 4))      # 第 3 個平日
    assert not m.is_rebalance_window(date(2026, 11, 5))
    assert m.next_rebalance(date(2026, 10, 6)) == "2026-11-02"
