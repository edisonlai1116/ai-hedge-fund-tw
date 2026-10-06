"""技術面：資料不足回 insufficient_data（不當 neutral 加權）、動能用複利、波動無方向、Hurst 不入總分。"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langchain_core")
from src.agents import technicals as t  # noqa: E402


def _df(n, drift=0.001, vol=0.02, seed=1):
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = pd.Series(100 * np.cumprod(1 + np.random.default_rng(seed).normal(drift, vol, n)), index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)


@pytest.mark.parametrize("fn,key", [
    (t.calculate_trend_signals, "trend"),
    (t.calculate_momentum_signals, "momentum"),
    (t.calculate_mean_reversion_signals, "mean_reversion"),
    (t.calculate_volatility_signals, "volatility"),
])
def test_insufficient_data_is_explicit(fn, key):
    out = fn(_df(t.MIN_BARS[key] - 1))
    assert out["signal"] == t.INSUFFICIENT
    assert out["score"] is None


def test_insufficient_components_are_excluded_not_neutral():
    bull = {"signal": "bullish", "confidence": 0.8, "score": 90.0}
    missing = {"signal": t.INSUFFICIENT, "confidence": 0.0, "score": None}
    out = t.weighted_signal_combination({"trend": bull, "momentum": missing, "mean_reversion": missing}, t.TECHNICAL_WEIGHTS)
    assert out["score"] == 90.0                       # 沒被「neutral 50」稀釋
    assert out["components_used"] == ["trend"]
    none = t.weighted_signal_combination({"trend": missing, "momentum": missing}, t.TECHNICAL_WEIGHTS)
    assert none["signal"] == t.INSUFFICIENT


def test_momentum_is_compounded():
    close = pd.Series([100.0, 110.0, 99.0])           # +10% 再 -10%
    assert t.compounded_return(close, 2) == pytest.approx(-0.01)   # 加總會錯誤地得到 0
    assert t.compounded_return(close, 5) is None


def test_volatility_has_no_direction():
    calm, wild = _df(300, vol=0.005), _df(300, vol=0.06, seed=2)
    for d in (calm, wild):
        v = t.calculate_volatility_signals(d)
        assert v["signal"] == "neutral" and v["directional"] is False
    assert "volatility" not in t.TECHNICAL_WEIGHTS


def test_hurst_is_experimental_and_not_weighted():
    out = t.calculate_stat_arb_signals(_df(300))
    assert out["experimental"] is True and out["score"] is None
    assert "stat_arb" not in t.TECHNICAL_WEIGHTS
