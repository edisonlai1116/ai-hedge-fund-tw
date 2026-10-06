"""爆量長紅點火偵測（detect_ignition）與趨勢回測（無固定停利）的離線單元測試。"""
import numpy as np
import pandas as pd

from src.simple_signal import detect_ignition


def _frame(closes, volumes):
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.01, "Low": closes * 0.99, "Close": closes, "Volume": volumes},
        index=idx,
    )


def test_ignition_today():
    closes = [100.0] * 30 + [109.0]
    vols = [1_000_000] * 30 + [1_800_000]
    out = detect_ignition(_frame(closes, vols))
    assert out["ignition_days_ago"] == 0
    assert out["ignition_gain_pct"] >= 8.9


def test_no_ignition_without_volume():
    closes = [100.0] * 30 + [109.0]
    vols = [1_000_000] * 31
    assert detect_ignition(_frame(closes, vols))["ignition_days_ago"] is None


def test_ignition_two_days_ago_holding_gains():
    closes = [100.0] * 30 + [110.0, 108.0, 107.0]
    vols = [1_000_000] * 30 + [2_000_000, 1_000_000, 1_000_000]
    assert detect_ignition(_frame(closes, vols))["ignition_days_ago"] == 2


def test_failed_ignition_gave_back_gains():
    closes = [100.0] * 30 + [110.0, 103.0, 102.0]
    vols = [1_000_000] * 30 + [2_000_000, 1_000_000, 1_000_000]
    assert detect_ignition(_frame(closes, vols))["ignition_days_ago"] is None
