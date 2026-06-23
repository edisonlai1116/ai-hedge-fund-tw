"""Unit tests for src/data/chip_flow.py — all pure functions, no network."""
import pytest
from src.data.chip_flow import (
    score_from_net_ratio,
    score_from_streak,
    blend_tw_score,
    chip_label,
    cmf_to_score,
    chip_flow_score,
    _fmt,
    _streak_label,
)


# ── score_from_net_ratio ────────────────────────────────────────────
class TestScoreFromNetRatio:
    def test_neutral(self):
        assert score_from_net_ratio(0.0) == 50

    def test_strong_buy(self):
        assert score_from_net_ratio(10.0) == 100

    def test_strong_sell(self):
        assert score_from_net_ratio(-10.0) == 0

    def test_clamped_above(self):
        assert score_from_net_ratio(999.0) == 100

    def test_clamped_below(self):
        assert score_from_net_ratio(-999.0) == 0

    def test_partial_buy(self):
        # +5% → 75
        assert score_from_net_ratio(5.0) == 75


# ── score_from_streak ───────────────────────────────────────────────
class TestScoreFromStreak:
    def test_neutral(self):
        assert score_from_streak(0) == 50

    def test_5_consecutive_buys(self):
        assert score_from_streak(5) == 90

    def test_5_consecutive_sells(self):
        assert score_from_streak(-5) == 10

    def test_clamped(self):
        assert score_from_streak(100) == 100
        assert score_from_streak(-100) == 0


# ── blend_tw_score ─────────────────────────────────────────────────
class TestBlendTwScore:
    def test_both_neutral(self):
        assert blend_tw_score(0.0, 0) == 50

    def test_both_bullish(self):
        s = blend_tw_score(10.0, 5)
        assert s >= 90

    def test_conflicting(self):
        s = blend_tw_score(10.0, -5)
        assert 45 <= s <= 55


# ── chip_label ─────────────────────────────────────────────────────
class TestChipLabel:
    def test_bullish(self):   assert chip_label(70) == "法人偏多"
    def test_bearish(self):   assert chip_label(30) == "法人偏空"
    def test_neutral(self):   assert chip_label(50) == "法人中性"
    def test_boundary_bull(self): assert chip_label(65) == "法人偏多"
    def test_boundary_bear(self): assert chip_label(44) == "法人偏空"


# ── cmf_to_score ───────────────────────────────────────────────────
class TestCmfToScore:
    def test_zero(self):   assert cmf_to_score(0.0) == 50
    def test_plus(self):   assert cmf_to_score(0.1) == 75
    def test_minus(self):  assert cmf_to_score(-0.1) == 25
    def test_max(self):    assert cmf_to_score(1.0) == 100
    def test_min(self):    assert cmf_to_score(-1.0) == 0


# ── T86 table lookup (offline) ─────────────────────────────────────
class TestChipFlowScoreTW:
    SAMPLE_TABLE = {
        "2330": {"foreign_net": 5_000_000, "trust_net": 1_000_000,
                 "dealer_net": -200_000, "total_net": 5_800_000},
        "2317": {"foreign_net": -3_000_000, "trust_net": -500_000,
                 "dealer_net": 0, "total_net": -3_500_000},
    }

    def test_tw_bullish_returns_dict(self):
        result = chip_flow_score("2330.TW", df=None, tw_table=self.SAMPLE_TABLE)
        assert result is not None
        assert result["source"] == "TWSE T86"
        assert 0 <= result["score"] <= 100
        assert result["foreign_net"] == 5_000_000

    def test_tw_bearish(self):
        result = chip_flow_score("2317.TW", df=None, tw_table=self.SAMPLE_TABLE)
        assert result is not None
        assert result["score"] < 50

    def test_tw_missing_code_returns_none(self):
        result = chip_flow_score("9999.TW", df=None, tw_table=self.SAMPLE_TABLE)
        assert result is None

    def test_tw_empty_table_returns_none(self):
        result = chip_flow_score("2330.TW", df=None, tw_table={})
        assert result is None


# ── US stock uses CMF (needs pandas DataFrame) ─────────────────────
class TestChipFlowScoreUS:
    def _make_df(self, trend: str = "up"):
        """Create synthetic OHLCV where close is near high (up) or near low (down)
        so CMF has a clear directional signal."""
        import pandas as pd, numpy as np
        n = 50
        idx = pd.date_range("2025-01-01", periods=n)
        base = np.linspace(100, 130, n) if trend == "up" else np.linspace(130, 100, n)
        if trend == "up":
            # close near top of range → positive money flow multiplier
            high  = base * 1.02
            low   = base * 0.98
            close = base * 1.015   # close near high
        else:
            # close near bottom of range → negative money flow multiplier
            high  = base * 1.02
            low   = base * 0.98
            close = base * 0.985   # close near low
        vol = np.full(n, 1_000_000.0)
        return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": vol}, index=idx)

    def test_us_uptrend_bullish(self):
        df = self._make_df("up")
        result = chip_flow_score("NVDA", df=df, tw_table=None)
        assert result is not None
        assert result["source"] == "CMF"
        assert result["score"] > 50

    def test_us_downtrend_bearish(self):
        df = self._make_df("down")
        result = chip_flow_score("NVDA", df=df, tw_table=None)
        assert result is not None
        assert result["score"] < 50

    def test_us_no_df_returns_none(self):
        result = chip_flow_score("NVDA", df=None, tw_table=None)
        assert result is None


# ── formatting helpers ─────────────────────────────────────────────
class TestFormatHelpers:
    def test_fmt_positive(self):
        assert _fmt(3_456_789) == "+3,456張"

    def test_fmt_negative(self):
        s = _fmt(-2_000_000)
        assert "2,000張" in s

    def test_streak_label_consecutive_buy(self):
        assert _streak_label(3) == "連買3日"

    def test_streak_label_sell(self):
        assert _streak_label(-1) == "賣超"

    def test_streak_label_neutral(self):
        assert _streak_label(0) == "中性"
