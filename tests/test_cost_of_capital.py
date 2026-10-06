"""WACC 修正：信用利差有上下限、不再因公式錯誤被頂到上限、beta 由價格估計。"""
import numpy as np
import pandas as pd
import pytest

from src.agents import cost_of_capital as cc


def test_coverage_five_no_longer_gives_200pct_cost_of_debt():
    # 舊公式：rf + 10/5 = rf + 200%
    cod = cc.cost_of_debt(0.043, 5.0)
    assert 0.043 + 0.01 <= cod <= 0.043 + 0.02


@pytest.mark.parametrize("cov", [-5, 0, 0.3, 1, 2, 3, 5, 10, 100, None, float("nan")])
def test_credit_spread_bounded(cov):
    s = cc.credit_spread_by_interest_coverage(cov)
    assert cc.MIN_CREDIT_SPREAD <= s <= cc.MAX_CREDIT_SPREAD


def test_spread_monotonic_in_coverage():
    covs = [0.5, 1, 1.5, 2, 3, 5, 8, 12]
    spreads = [cc.credit_spread_by_interest_coverage(c) for c in covs]
    assert spreads == sorted(spreads, reverse=True)


def test_typical_companies_not_capped():
    for cov, beta in [(2.0, 1.2), (5.0, 1.0), (20.0, 1.6)]:
        w = cc.calculate_wacc(market_cap=100e9, total_debt=30e9, cash=5e9, interest_coverage=cov,
                              beta=beta, risk_free_rate=0.043)
        assert not w["bounded"]
        assert 0.06 < w["wacc"] < 0.16


def test_beta_used_and_flagged():
    w = cc.calculate_wacc(market_cap=1e9, total_debt=0, cash=0, interest_coverage=10, beta=1.8, risk_free_rate=0.04,
                          market_risk_premium=0.05)
    assert w["cost_of_equity"] == pytest.approx(0.04 + 1.8 * 0.05)
    assert w["beta_is_default"] is False
    d = cc.calculate_wacc(market_cap=1e9, total_debt=0, cash=0, interest_coverage=10, beta=None, risk_free_rate=0.04)
    assert d["beta_is_default"] is True


def test_estimate_beta_recovers_true_beta():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-01", periods=600)
    m = rng.normal(0.0004, 0.01, 600)
    s = 1.7 * m + rng.normal(0, 0.004, 600)
    beta = cc.estimate_beta(pd.Series(100 * np.cumprod(1 + s), idx), pd.Series(100 * np.cumprod(1 + m), idx))
    assert beta == pytest.approx(1.7, abs=0.2)
    assert cc.estimate_beta(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) is None
