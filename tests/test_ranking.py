"""Cross-sectional ranking 系統的離線測試（不連網）。"""
import numpy as np
import pandas as pd
import pytest

from src.ranking import features as fx
from src.ranking import scoring as sc
from src.ranking.regime import classify_regime
from src.ranking.shocks import detect_shocks, fundamental_damage, shock_verdict


def _ohlcv(close, vol=1e6):
    c = pd.Series(close, index=pd.bdate_range("2024-01-01", periods=len(close)), dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": vol})


# ---------------- 分數與狀態 ----------------
def test_quality_name_that_is_overheated_is_not_buy_now():
    """規格範例：Quality 95、Growth 98、Catalyst 95、Overextension 90 → 不是直接 BUY。"""
    comp = {"momentum_rank": 90, "quality": 95, "growth": 98, "catalyst": 95, "earnings_acceleration": 90,
            "valuation": 40, "ai_exposure": 90}
    opp = sc.opportunity_score(comp, overextension=90, weights=sc.regime_weights(None))
    st = sc.decide_status(opp["score"], 90, 95, 50, 40, 90, held=False)
    assert st["status"] in ("BUY_ON_PULLBACK", "HOLD", "WAIT")
    assert "WAIT_ENTRY" in st["flags"]
    held = sc.decide_status(opp["score"], 90, 95, 50, 40, 90, held=True)
    assert held["status"] in ("HOLD_CORE", "PARTIAL_PROFIT", "HOLD")


def test_missing_modules_are_excluded_not_neutral():
    w = sc.regime_weights(None)
    full = sc.opportunity_score({"momentum_rank": 90}, None, w)
    assert full["score"] == 90.0            # 其他模組缺資料不會被當 50 拉低
    assert full["coverage"] < 1
    assert sc.opportunity_score({}, None, w)["score"] is None


def test_big_drop_with_low_damage_is_oversold_not_sell():
    idx = pd.bdate_range("2024-01-01", periods=10)
    stock = pd.Series([0.0] * 9 + [-0.10], index=idx)
    spy = pd.Series([0.0] * 9 + [-0.025], index=idx)
    peers = pd.Series([0.0] * 9 + [-0.07], index=idx)
    shocks = detect_shocks(stock, spy, peers, beta=1.5, overextension_before=40)
    assert shocks and set(shocks[0]["causes"]) >= {"macro", "industry"}
    dmg = fundamental_damage(shocks[0], catalyst_events=[], eps_revision_90d=0.02, latest_surprise=5, recovered_pct=0.01)
    assert dmg["fundamental_damage_score"] < 40
    assert shock_verdict(shocks[0], dmg) == "POTENTIAL_OVERSOLD"
    st = sc.decide_status(52, 10, 70, 60, 50, 40, held=False, damage=dmg["fundamental_damage_score"], oversold=True)
    assert st["status"] != "SELL" and "POTENTIAL_OVERSOLD" in st["flags"]


def test_company_specific_drop_with_bad_news_is_thesis_risk():
    idx = pd.bdate_range("2024-01-01", periods=5)
    shocks = detect_shocks(pd.Series([0, 0, 0, 0, -0.15], index=idx), pd.Series([0, 0, 0, 0, 0.002], index=idx),
                           pd.Series([0, 0, 0, 0, -0.005], index=idx), 1.0, 20)
    assert "company_specific" in shocks[0]["causes"]
    ev = [{"direction": -1, "category": "guidance", "age_days": 1}]
    dmg = fundamental_damage(shocks[0], ev, eps_revision_90d=-0.10, latest_surprise=-12, recovered_pct=-0.05)
    assert dmg["structural"] and shock_verdict(shocks[0], dmg) == "THESIS_AT_RISK"


# ---------------- 特徵 ----------------
def test_features_compounded_and_point_in_time():
    pm = {"A": _ohlcv(np.linspace(100, 200, 300)), "B": _ohlcv(np.linspace(100, 90, 300))}
    spy = pm["A"]["Close"] * 0 + 100
    F = fx.compute_features(pm, {"SPY": spy, "QQQ": spy})
    c = pm["A"]["Close"]
    assert F["ret_20"]["A"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-21] - 1)
    # 截斷未來資料不影響過去的特徵值（point-in-time）
    F2 = fx.compute_features({k: v.iloc[:200] for k, v in pm.items()}, {"SPY": spy.iloc[:200], "QQQ": spy.iloc[:200]})
    assert F2["ret_60"]["A"].iloc[-1] == pytest.approx(F["ret_60"]["A"].iloc[199])


def test_overextension_is_volatility_normalized():
    rng = np.random.default_rng(0)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    hot = base.copy()
    hot[-10:] = hot[-11] * np.cumprod(np.full(10, 1.03))       # 最後 10 天每天 +3%
    pm = {"HOT": _ohlcv(hot), "CALM": _ohlcv(base)}
    F = fx.compute_features(pm, {"SPY": pm["CALM"]["Close"]})
    ox = fx.overextension_score(F).iloc[-1]
    assert ox["HOT"] > 70 and ox["CALM"] < ox["HOT"]


# ---------------- 進場區、輪動、部位 ----------------
def test_entry_zones_are_below_price_and_ordered():
    rng = np.random.default_rng(1)
    df = _ohlcv(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400)))
    z = sc.entry_zones(df)
    px = z["price"]
    assert z["zone_1"]["high"] <= px * 1.01
    assert z["zone_1"]["low"] > z["zone_2"]["low"] > z["zone_3"]["low"]
    assert z["avoid_above"] >= z["zone_1"]["high"]


def _row(t, opp, status, held=False, vol=0.4, beta=1.0, theme="X", **scores):
    return {"ticker": t, "opportunity": opp, "status": status, "held": held, "vol_63": vol, "beta": beta, "theme": theme,
            "scores": {"valuation": 50, "catalyst": 50, "overextension": 20, **scores}, "coverage": 1.0, "downside_risk": vol}


def test_rotation_pairs_sell_weak_hold_for_strong_candidate():
    rows = [_row("A", 30, "HOLD", held=True), _row("B", 85, "BUY_NOW")]
    rot = sc.rotation_suggestions(rows)
    assert rot and rot[0]["sell"] == "A" and rot[0]["buy"] == "B"
    assert rot[0]["type"] in ("PARTIAL_ROTATION", "FULL_ROTATION")
    assert sc.rotation_suggestions([_row("A", 80, "HOLD", held=True), _row("B", 82, "BUY_NOW")]) == []


def test_position_sizes_respect_caps():
    rows = [_row(f"S{i}", 90, "BUY_NOW", theme="AI Compute", beta=2.0) for i in range(6)] + [_row("U", 70, "BUY_NOW", theme="Power")]
    w = sc.position_sizes(rows, {"exposure": 1.0, "max_high_beta_weight": 0.08})
    assert max(w.values()) <= 0.12 + 1e-9
    assert sum(v for k, v in w.items() if k.startswith("S")) <= 0.35 + 1e-6
    assert all(v <= 0.08 + 1e-9 for k, v in w.items() if k.startswith("S"))


# ---------------- 市場狀態 ----------------
def test_regime_risk_off_and_bull():
    idx = pd.bdate_range("2023-01-01", periods=300)
    up = pd.Series(np.linspace(100, 160, 300), index=idx)
    down = pd.Series(np.linspace(160, 100, 300), index=idx)
    vix_low = pd.Series(14.0, index=idx)
    assert classify_regime({"SPY": down, "QQQ": down, "VIX": vix_low})["regime"] == "RISK_OFF"
    qqq = pd.Series(np.linspace(100, 190, 300), index=idx)
    assert classify_regime({"SPY": up, "QQQ": qqq, "VIX": vix_low})["regime"] == "BULL"
    assert classify_regime({"SPY": up.iloc[:50]})["regime"] == "insufficient_data"


# ---------------- Portfolio Manager ----------------
def test_pm_sells_before_buying_and_waits_for_zone():
    pytest.importorskip("langchain_core")
    from src.agents.portfolio_manager import decide_from_ranking
    table = [
        {"ticker": "B", "rank": 1, "price": 100.0, "opportunity": 85, "status": "BUY_NOW", "target_weight": 0.10,
         "position_shares": 0, "entry": {}, "narrative": {}},
        {"ticker": "C", "rank": 2, "price": 100.0, "opportunity": 70, "status": "BUY_ON_PULLBACK", "target_weight": 0.10,
         "position_shares": 0, "entry": {"zone_1": {"low": 90, "high": 95}}, "narrative": {}},
        {"ticker": "A", "rank": 3, "price": 50.0, "opportunity": 20, "status": "SELL", "target_weight": 0.0,
         "position_shares": 200, "entry": {}, "narrative": {"key_risk": "x"}},
    ]
    portfolio = {"cash": 0.0, "positions": {"A": {"long": 200, "short": 0}}}
    d = decide_from_ranking(table, {"rotations": []}, portfolio, {"A": 50.0, "B": 100.0, "C": 100.0},
                            {"A": 1000, "B": 1000, "C": 1000})
    assert d["A"].action == "sell" and d["A"].quantity == 200
    assert d["B"].action == "buy" and d["B"].quantity > 0          # 用賣 A 的錢買 B
    assert d["C"].action == "hold"                                  # 價格 100 不在 Zone 1（≤95）→ 等拉回


# ---------------- 訊號追蹤 ----------------
def test_tracking_logs_and_evaluates(tmp_path):
    from datetime import date
    from src.ranking import tracking as tr
    path = str(tmp_path / "log.jsonl")
    ranking = {"regime": {"regime": "BULL"}, "ranking": [
        {"ticker": "UP", "status": "BUY_NOW", "opportunity": 80, "price": 100.0, "scores": {"momentum_rank": 90}},
        {"ticker": "DN", "status": "SELL", "opportunity": 20, "price": 100.0, "scores": {"momentum_rank": 10}},
    ]}
    assert tr.log_signals(ranking, path, asof="2025-01-02") == 2
    assert tr.log_signals(ranking, path, asof="2025-01-02") == 0          # 同日不重複
    idx = pd.bdate_range("2024-12-01", periods=120)
    prices = {"UP": pd.Series(np.linspace(100, 160, 120), index=idx), "DN": pd.Series(np.linspace(100, 70, 120), index=idx),
              "SPY": pd.Series(np.linspace(100, 105, 120), index=idx)}
    out = tr.evaluate(path, today=date(2025, 4, 1), price_loader=lambda syms: dict(prices))
    assert out["signal_accuracy"]["BUY_NOW"]["20d"]["hit_rate_pct"] == 100.0
    assert out["signal_accuracy"]["SELL"]["20d"]["hit_rate_pct"] == 100.0
    assert out["signal_accuracy"]["BUY_NOW"]["20d"]["conclusive"] is False   # 樣本太少不下結論
    assert out["signal_accuracy"]["BUY_NOW"]["avg_mfe_60d_pct"] > 0


def test_news_relevance_filter():
    from src.ranking.catalysts import _relevant
    assert not _relevant("Crescent Energy (CRGY) Stock Could Be Above Fair Value", "BE", "Bloom Energy Corporation")
    assert _relevant("Bloom Energy soars on new order", "BE", "Bloom Energy Corporation")
    assert _relevant("Why Vistra Stock Popped", "VST", "Vistra Corp.")


def test_above_avoid_forces_pullback_and_mid_score_buys_on_pullback():
    assert sc.decide_status(80, 20, 70, 50, 50, 60, held=False, above_avoid=True)["status"] == "BUY_ON_PULLBACK"
    assert sc.decide_status(80, 20, 70, 50, 50, 60, held=False)["status"] == "BUY_NOW"
    assert sc.decide_status(60, 20, 70, 50, 50, 60, held=False)["status"] == "BUY_ON_PULLBACK"
    assert sc.decide_status(45, 20, 70, 50, 50, 60, held=False)["status"] == "WAIT"
