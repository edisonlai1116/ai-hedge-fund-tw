"""Decision Logic Consistency：Opportunity / Entry / Overextension / Risk / Action / Sizing 彼此不矛盾。

P9  Decision Consistency Matrix（對 decide_action 的全組合網格 + 正式流程）
P10 Regression：用「與真實案例特徵相同」的合成價格路徑跑正式的 features → decision_pipeline（不連網、無個股特例）。
    案例名稱只是標籤；規則裡沒有任何 ticker 判斷。
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.ranking import features as fx
from src.ranking import scoring as sc
from src.ranking.engine import decision_pipeline
from src.ranking.themes import theme_exposures

BUYS = ("BUY_NOW",)
DRAWDOWN_TYPES = ("NONE", "FUNDAMENTAL_DISCOUNT", "TEMPORARY_SHOCK", "VALUATION_RESET", "SPECULATIVE_DE_RATING",
                  "FUNDAMENTAL_DAMAGE", "UNKNOWN")


# =============================================================================
# 合成價格與正式流程
# =============================================================================
def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    c = pd.Series(close, index=pd.bdate_range("2022-01-03", periods=len(close)), dtype=float)
    return pd.DataFrame({"Open": c.shift(1).fillna(c.iloc[0]), "High": c * 1.012, "Low": c * 0.988, "Close": c, "Volume": 1e6})


def _walk(n, drift, vol, seed, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(drift, vol, n))


def _then(path, rets):
    return np.concatenate([path, path[-1] * np.cumprod(1 + np.asarray(rets))])


def _rets(n, drift, vol, seed):
    return np.random.default_rng(seed).normal(drift, vol, n)


def _rally(n, total, noise, seed):
    rng = np.random.default_rng(seed)
    base = (1 + total) ** (1 / n) - 1
    r = base + rng.normal(0, noise, n)
    return r - (np.prod(1 + r) ** (1 / n) - 1 - base)   # 調整到總漲幅 ≈ total


def evaluate(close, fund, held=False, shock_view=None, fund_metrics=None, universe_rank=None):
    """synthetic close → 正式 features → 正式 decision_pipeline。fund：quality/growth/earnings_acceleration/valuation/ai_exposure/catalyst。"""
    n = len(close)
    peers = {f"P{i}": _walk(n, 0.0004, 0.02, 100 + i) for i in range(6)}
    pm = {"X": _ohlcv(close), **{k: _ohlcv(v) for k, v in peers.items()}}
    spy = _ohlcv(_walk(n, 0.0003, 0.01, 7))["Close"]
    F = fx.compute_features(pm, {"SPY": spy, "QQQ": spy})
    last = lambda df: fx.last_row(df)["X"]   # noqa: E731
    scores = {
        "momentum_rank": float(last(fx.pct_rank(F["sharpe_126"]))),
        "relative_strength": float(last(fx.relative_strength_score(F))),
        "technical": float(last(fx.technical_score(F))),
        "risk": float(last(fx.risk_score(F))),
        "overextension": float(last(fx.overextension_score(F))),
        **fund,
    }
    g = lambda k: float(last(F[k]))   # noqa: E731
    d = decision_pipeline(pm["X"], scores, None, dd52=g("drawdown_252"), ret_3y=g("ret_756"), dist_ma50=g("dist_ma50"),
                          shock_view=shock_view, fund_metrics=fund_metrics, days_to_earnings=None, held=held,
                          ret_20=g("ret_20"), universe_rank=universe_rank)
    d.update(scores=scores, r20=g("ret_20"), rsi=g("rsi14"), price=d["zones"]["price"])
    return d


def assert_consistent(d):
    """所有正式流程輸出都必須滿足的不變量。"""
    st, cond, z1 = d["action"], d["cond"], d["zones"]["zone_1"]
    nm = d["new_money"]
    ox = d["scores"]["overextension"]
    for a in (st, nm):
        if a["status"] == "BUY_NOW":
            assert cond["ok"], "BUY_NOW 但現價不滿足進場條件"
            assert d["price"] <= z1["high"] or cond["basis"] in ("ACCEPTABLE_ABOVE_BUY1", "BELOW_LOW_ENTRY_LINE"), \
                "BUY_NOW 且現價高於 Buy1，卻沒有明確標示可接受的理由"
            assert ox < sc.OX_EXTENDED, "BUY_NOW 但過熱 ≥ Extended"
            assert d["drawdown"]["type"] != "FUNDAMENTAL_DAMAGE", "BUY_NOW 但基本面受損"
        if a["status"] == "BUY_ON_PULLBACK":
            assert a["pullback_reason"] in ("ABOVE_ENTRY", "CONFIRMATION_REQUIRED", "COOLING_REQUIRED")
            if a["pullback_reason"] == "ABOVE_ENTRY":
                assert not cond["ok"]
            if a["pullback_reason"] == "COOLING_REQUIRED":
                assert ox >= sc.OX_EXTENDED
        if a["status"] == "WAIT" and ox >= sc.OX_EXTENDED:
            assert "不足" not in a["why"], "過熱造成的 WAIT 不能說成「分數不足」"


# =============================================================================
# P9 Decision Consistency Matrix（網格：所有輸入組合都不能出現矛盾）
# =============================================================================
GRID = list(itertools.product(
    (20, 50, 62, 70, 90),                 # opp
    (0, 30, 55, 75, 90),                  # overextension
    (None, 40, 80),                       # quality
    (30, 95),                             # risk
    (None, 20, 60),                       # valuation
    (False, True),                        # held
    (None, 10, 70),                       # damage
    (False, True),                        # entry_ok
    (False, True),                        # low_entry
    DRAWDOWN_TYPES,
))


def _decide(opp, ox, q, risk, val, held, dmg, ok, low, dt):
    return sc.decide_action(opp, ox, q, risk, val, 50, held, damage=dmg, oversold=False, entry_ok=ok, low_entry=low,
                            long_term_broken=False, growth=q, drawdown_type=dt)


def test_matrix_1_buy_now_requires_entry_condition():
    for opp, ox, q, risk, val, held, dmg, ok, low, dt in GRID:
        if not ok:
            assert _decide(opp, ox, q, risk, val, held, dmg, ok, low, dt)["status"] != "BUY_NOW"


def test_matrix_2_buy_now_never_extreme_or_extended():
    for opp, ox, q, risk, val, held, dmg, ok, low, dt in GRID:
        if ox >= sc.OX_EXTENDED:
            assert _decide(opp, ox, q, risk, val, held, dmg, ok, low, dt)["status"] != "BUY_NOW"


def test_matrix_3_buy_now_never_with_fundamental_damage():
    for opp, ox, q, risk, val, held, dmg, ok, low, dt in GRID:
        if dt == "FUNDAMENTAL_DAMAGE" or (dmg or 0) >= sc.DAMAGE_BLOCK:
            assert _decide(opp, ox, q, risk, val, held, dmg, ok, low, dt)["status"] not in ("BUY_NOW", "BUY_ON_PULLBACK")


def test_matrix_4_and_11_good_company_extremely_overextended_waits():
    st = sc.decide_action(85, 90, 85, 40, 40, 70, held=False, entry_ok=True, growth=85, drawdown_type="NONE")
    assert st["status"] == "WAIT" and st["state"] == "GOOD_BUT_OVEREXTENDED"
    assert "不足" not in st["why"] and "過熱" in st["why"]
    st = sc.decide_action(85, 60, 85, 40, 40, 70, held=False, entry_ok=True, growth=85, drawdown_type="NONE")
    assert st["status"] == "BUY_ON_PULLBACK" and st["state"] == "GOOD_BUT_OVEREXTENDED"


def test_matrix_5_buy_on_pullback_when_price_above_entry():
    st = sc.decide_action(80, 10, 70, 40, 50, 60, held=False, entry_ok=False, drawdown_type="NONE")
    assert st["status"] == "BUY_ON_PULLBACK" and st["pullback_reason"] == "ABOVE_ENTRY"


def test_matrix_buy_on_pullback_always_has_reason_and_wait_never_buys():
    for opp, ox, q, risk, val, held, dmg, ok, low, dt in GRID:
        st = _decide(opp, ox, q, risk, val, held, dmg, ok, low, dt)
        if st["status"] == "BUY_ON_PULLBACK":
            assert st["pullback_reason"] in ("ABOVE_ENTRY", "CONFIRMATION_REQUIRED", "COOLING_REQUIRED")
            if st["pullback_reason"] == "ABOVE_ENTRY":
                assert not ok
            if st["pullback_reason"] == "COOLING_REQUIRED":
                assert ox >= sc.OX_EXTENDED
        if st["status"] == "WAIT" and ox >= sc.OX_EXTENDED:
            assert "不足" not in st["why"]


def test_matrix_6_hold_core_for_quality_holder_is_not_a_new_money_buy():
    st = sc.decide_action(60, 55, 85, 40, 40, 60, held=True, entry_ok=False, growth=80, drawdown_type="NONE")
    assert st["status"] == "HOLD_CORE"
    new_money = sc.decide_action(60, 55, 85, 40, 40, 60, held=False, entry_ok=False, growth=80, drawdown_type="NONE")
    assert new_money["status"] != "BUY_NOW"


def test_matrix_7_drawdown_with_low_damage_can_buy():
    for dt in sc.BUYABLE_DRAWDOWNS:
        st = sc.decide_action(50, 10, 70, 50, 55, 30, held=False, damage=10, entry_ok=True, low_entry=True, drawdown_type=dt)
        assert st["status"] == "BUY_NOW"


def test_matrix_8_drawdown_with_damage_cannot_buy_now():
    st = sc.decide_action(80, 5, 70, 50, 60, 30, held=False, damage=70, entry_ok=True, low_entry=True,
                          drawdown_type="FUNDAMENTAL_DAMAGE")
    assert st["status"] == "WAIT" and "THESIS_AT_RISK" in st["flags"]
    for dt in ("SPECULATIVE_DE_RATING", "UNKNOWN", "VALUATION_RESET"):
        st = sc.decide_action(80, 5, 70, 50, 60, 30, held=False, damage=10, entry_ok=True, low_entry=True, drawdown_type=dt)
        assert st["status"] != "BUY_NOW"


def _sz_row(t, theme, opp=80):
    return {"ticker": t, "opportunity": opp, "status": "BUY_NOW", "vol_63": 0.4, "downside_risk": 0.3, "beta": 1.0,
            "coverage": 1.0, "theme": theme, "scores": {"overextension": 10}}


def test_matrix_9_theme_cap_reduces_size():
    power = ["VST", "CEG", "NRG", "OKLO", "BE"]
    rows = [_sz_row(t, "Power") for t in power] + [_sz_row("MSFT", "AI Platform", 60)]
    exp = {t: theme_exposures(t) for t in power + ["MSFT"]}
    w = sc.position_sizes(rows, {"exposure": 1.0}, exposures=exp, max_single=0.5)
    te = sc.theme_exposure(w, exp)
    assert te["Nuclear"] <= sc.THEME_CAP + 1e-6 and te["AI Data Center Power"] <= sc.THEME_CAP + 1e-6
    uncapped = sc.position_sizes(rows, {"exposure": 1.0}, exposures=exp, max_single=0.5, theme_cap=10.0)
    assert sum(w[t] for t in power if t in w) < sum(uncapped[t] for t in power)
    # 五檔電力/核能股不被視為獨立：曝險 = Σ 權重 × 曝險
    assert sc.theme_exposure({"VST": 0.1, "CEG": 0.1}, exp)["Nuclear"] == pytest.approx(0.2)


def test_matrix_10_portfolio_exposure_never_changes_stock_scores():
    rows = [_sz_row(t, "Power") for t in ("VST", "CEG", "NRG")]
    before = [dict(r, scores=dict(r["scores"])) for r in rows]
    sc.position_sizes(rows, {"exposure": 1.0}, exposures={t: theme_exposures(t) for t in ("VST", "CEG", "NRG")},
                      theme_cap=0.05)
    assert rows == before
    close = _then(_walk(800, 0.001, 0.02, 1), _rally(20, 0.05, 0.01, 2))
    fund = {"quality": 70, "growth": 70, "earnings_acceleration": 60, "valuation": 50, "ai_exposure": 60}
    a, b = evaluate(close, dict(fund), held=False), evaluate(close, dict(fund), held=True)
    assert a["opp"]["score"] == b["opp"]["score"] and a["entry_score"] == b["entry_score"]
    assert a["scores"]["overextension"] == b["scores"]["overextension"] and a["scores"]["risk"] == b["scores"]["risk"]


def test_matrix_12_high_volatility_rally_is_not_less_overextended():
    """同一段歷史、同樣最後 20 日 +30% 的急漲；只差在急漲前 43 天的波動（±4% vs ±0.5% 鋸齒、終點相同）。
    高波動的那檔過熱分數不能比較低（舊版用波動單位衡量，高波動股反而被判「不熱」）。"""
    hist = _walk(740, 0.0008, 0.015, 3)
    zig = lambda a: np.tile([1 + a, 1 / (1 + a)], 22)[:43]   # noqa: E731  終點回到原水位
    rally = _rally(20, 0.30, 0.004, 4)
    calm = _then(hist, zig(0.005) - 1)
    wild = _then(hist, zig(0.04) - 1)
    calm, wild = _then(calm, rally), _then(wild, rally)
    pm = {"CALM": _ohlcv(calm), "WILD": _ohlcv(wild), "P": _ohlcv(_walk(len(calm), 0.0004, 0.02, 9))}
    F = fx.compute_features(pm, {"SPY": pm["P"]["Close"]})
    assert fx.last_row(F["vol_63"])["WILD"] > 2 * fx.last_row(F["vol_63"])["CALM"]
    ox = fx.last_row(fx.overextension_score(F))
    assert ox["WILD"] >= ox["CALM"] - 1e-9
    assert fx.overextension_tier(ox["WILD"]) in ("Extended", "Overheated", "Extreme")


def test_overextension_tiers_and_calibration_monotonic():
    assert fx.overextension_tier(10) == "Normal" and fx.overextension_tier(30) == "Warm"
    assert fx.overextension_tier(55) == "Extended" and fx.overextension_tier(75) == "Overheated"
    assert fx.overextension_tier(90) == "Extreme"
    xs = [p[0] for p in fx.OVEREXTENSION_CALIBRATION]
    ys = [p[1] for p in fx.OVEREXTENSION_CALIBRATION]
    assert xs == sorted(xs) and ys == sorted(ys)


def test_entry_condition_explicit_acceptable_above_buy1():
    zones = {"zone_1": {"low": 280, "high": 290}, "atr": 10.0, "avoid_above": 400}
    assert not sc.entry_condition(303, zones, 10)["ok"]                         # 規格範例：303 vs Buy1 280–290
    ok = sc.entry_condition(295, zones, 10)
    assert ok["ok"] and ok["basis"] == "ACCEPTABLE_ABOVE_BUY1" and "acceptable" in ok["reason"]
    assert not sc.entry_condition(295, zones, 40)["ok"]                          # 偏熱就不給「可接受」
    assert sc.entry_condition(285, zones, 80)["basis"] == "IN_BUY1"
    assert sc.entry_condition(303, zones, 10, low_entry_price=310)["basis"] == "BELOW_LOW_ENTRY_LINE"


def test_entry_zones_strictly_ordered_even_near_52w_low():
    """Buy1 > Buy2 > Deep（舊版 Deep 用 52 週低點後備，可能高於 Buy2，例：跌到接近一年低點的股票）。"""
    for seed in range(30):
        close = _then(_walk(500, 0.002, 0.04, seed), _rets(150, -0.006, 0.04, seed + 1000))
        z = sc.entry_zones(_ohlcv(close))
        assert z["zone_1"]["low"] > z["zone_2"]["high"] - 1e-9 or z["zone_1"]["low"] > z["zone_2"]["low"]
        assert z["zone_2"]["low"] > z["zone_3"]["high"] - 1e-9, (seed, z["zone_2"], z["zone_3"])


def test_opportunity_excludes_entry_timing():
    """Opportunity 不含低檔深度（進場時機）——同一家公司跌 40% 與否，Opportunity 只隨動能排名等變化，不因「跌深」加分。"""
    assert "low_entry" not in sc.BASE_WEIGHTS and "overextension" not in sc.BASE_WEIGHTS
    w = sc.regime_weights(None)
    o = sc.opportunity_score({"momentum_rank": 50, "quality": 80}, None, w)["score"]
    assert o == sc.opportunity_score({"momentum_rank": 50, "quality": 80, "low_entry": 100}, None, w)["score"]


# =============================================================================
# P10 / P6 Regression（合成路徑重現案例特徵 → 正式流程）
# =============================================================================
STRONG = {"quality": 78, "growth": 85, "earnings_acceleration": 75, "valuation": 40, "ai_exposure": 80, "catalyst": 60}
PROFITABLE = {"operating_margin": 0.25, "fcf_margin": 0.15, "revenue": 5e9, "net_debt_to_ebitda": 0.5, "eps_revision_90d": 0.04}
PRE_REVENUE = {"operating_margin": -60.0, "fcf_margin": -190.0, "revenue": 0.0, "net_debt_to_ebitda": None, "eps_revision_90d": 0.0}


def test_regression_mrvl_like_good_but_overextended():
    """3 年上升趨勢 + 20 日 +26%、RSI ~70、技術強 → GOOD_BUT_OVEREXTENDED，WAIT / BUY_ON_PULLBACK，不是 BUY_NOW。"""
    close = _then(_walk(830, 0.0012, 0.028, 11), _rally(20, 0.26, 0.028, 12))
    d = evaluate(close, dict(STRONG), fund_metrics=PROFITABLE, universe_rank=(14, 102))
    assert 0.22 <= d["r20"] <= 0.30 and 60 <= d["rsi"] <= 80          # 夾具確實是案例特徵
    assert d["scores"]["overextension"] >= sc.OX_EXTENDED
    assert d["action"]["state"] == "GOOD_BUT_OVEREXTENDED"
    assert d["action"]["status"] in ("WAIT", "BUY_ON_PULLBACK")
    assert "不足" not in d["action"]["why"]
    assert_consistent(d)


def test_regression_wdc_like_drawdown_fundamentals_intact():
    """長線贏家跌 ~45%、近期大盤/產業拖累、基本面完好 → TEMPORARY_SHOCK/FUNDAMENTAL_DISCOUNT，BUY_NOW 或 BUY_ON_PULLBACK。"""
    close = _then(_walk(600, 0.0030, 0.02, 21), _rets(220, -0.0027, 0.025, 22))
    shock = {"shock": {"causes": ["macro", "industry"]}, "fundamental_damage_score": 10, "verdict": "POTENTIAL_OVERSOLD"}
    d = evaluate(close, {"quality": 80, "growth": 75, "earnings_acceleration": 70, "valuation": 55, "ai_exposure": 55},
                 shock_view=shock, fund_metrics=PROFITABLE, universe_rank=(59, 102))
    assert d["is_low"]
    assert d["drawdown"]["type"] in ("TEMPORARY_SHOCK", "FUNDAMENTAL_DISCOUNT")
    assert d["action"]["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")
    assert_consistent(d)


def test_regression_stx_like_company_specific_drop_fundamentals_strong():
    """公司特有大跌、關鍵字負面新聞（受損 50 只來自價格/新聞），但財報動能強、預估上修 → 不是 FUNDAMENTAL_DAMAGE。"""
    close = _then(_walk(650, 0.0025, 0.02, 31), _rets(170, -0.0019, 0.022, 32))
    shock = {"shock": {"causes": ["company_specific"]}, "fundamental_damage_score": 50, "verdict": None}
    d = evaluate(close, {"quality": 81, "growth": 83, "earnings_acceleration": 87, "valuation": 30, "ai_exposure": 55},
                 shock_view=shock, fund_metrics=PROFITABLE, universe_rank=(25, 102))
    assert d["drawdown"]["type"] != "FUNDAMENTAL_DAMAGE"
    assert d["action"]["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")
    assert_consistent(d)


def test_regression_avgo_like_quality_holder_hold_core():
    close = _walk(850, 0.0010, 0.018, 41)
    d = evaluate(close, {"quality": 88, "growth": 75, "earnings_acceleration": 70, "valuation": 45, "ai_exposure": 80},
                 held=True, fund_metrics=PROFITABLE, universe_rank=(40, 102))
    assert d["action"]["status"] == "HOLD_CORE"
    assert_consistent(d)


def test_regression_lite_like_hot_growth_expensive_not_auto_buy():
    """成長極佳、估值高、動能強且接近高點 → 不自動 BUY_NOW；持有者 HOLD / HOLD_CORE / PARTIAL_PROFIT。"""
    close = _then(_walk(780, 0.0015, 0.03, 51), _rally(50, 0.45, 0.02, 52))
    fund = {"quality": 55, "growth": 98, "earnings_acceleration": 70, "valuation": 20, "ai_exposure": 85}
    d = evaluate(close, dict(fund), fund_metrics=PROFITABLE)
    assert d["action"]["status"] != "BUY_NOW"
    assert d["action"]["status"] in ("WAIT", "BUY_ON_PULLBACK", "HOLD")
    h = evaluate(close, dict(fund), held=True, fund_metrics=PROFITABLE)
    assert h["action"]["status"] in ("HOLD", "HOLD_CORE", "PARTIAL_PROFIT")
    assert_consistent(d)
    assert_consistent(h)


def test_regression_oklo_scenario_a_speculative_derating_not_buy_now():
    """先大漲、再自高點 -70%、RSI 低、超賣，核心專案未惡化（無受損證據）→ SPECULATIVE_DE_RATING，不得只靠超賣 BUY_NOW。"""
    close = _then(_walk(600, 0.0080, 0.05, 61), _rets(230, -0.0045, 0.05, 62))
    shock = {"shock": {"causes": ["unexplained"]}, "fundamental_damage_score": 15, "verdict": "POTENTIAL_OVERSOLD"}
    d = evaluate(close, {"quality": 10, "growth": 60, "earnings_acceleration": 45, "valuation": 0, "ai_exposure": 70},
                 shock_view=shock, fund_metrics=PRE_REVENUE, universe_rank=(96, 102))
    assert d["is_low"] and d["low_score"] >= 80                       # 跌很多
    assert d["drawdown"]["type"] == "SPECULATIVE_DE_RATING"           # 但不是便宜
    assert d["action"]["status"] != "BUY_NOW"
    assert d["scores"]["fundamental_uncertainty"] >= 70
    assert_consistent(d)


def test_regression_oklo_scenario_b_thesis_deterioration_fundamental_damage():
    close = _then(_walk(600, 0.0080, 0.05, 61), _rets(230, -0.0045, 0.05, 62))
    shock = {"shock": {"causes": ["company_specific"]}, "fundamental_damage_score": 80, "verdict": "THESIS_AT_RISK"}
    d = evaluate(close, {"quality": 10, "growth": 30, "earnings_acceleration": 20, "valuation": 0, "ai_exposure": 70},
                 shock_view=shock, fund_metrics={**PRE_REVENUE, "eps_revision_90d": -0.15}, universe_rank=(96, 102))
    assert d["drawdown"]["type"] == "FUNDAMENTAL_DAMAGE"
    assert d["action"]["status"] not in ("BUY_NOW", "BUY_ON_PULLBACK")
    held = evaluate(close, {"quality": 10, "growth": 30, "earnings_acceleration": 20, "valuation": 0, "ai_exposure": 70},
                    held=True, shock_view=shock, fund_metrics={**PRE_REVENUE, "eps_revision_90d": -0.15}, universe_rank=(96, 102))
    assert held["action"]["status"] == "SELL"
    assert_consistent(d)


def test_regression_oklo_scenario_c_extreme_rally_good_but_overextended():
    close = _then(_walk(800, 0.0020, 0.05, 71), _rally(20, 0.70, 0.03, 72))
    d = evaluate(close, {"quality": 10, "growth": 90, "earnings_acceleration": 60, "valuation": 5, "ai_exposure": 90,
                         "catalyst": 90}, fund_metrics=PRE_REVENUE)
    assert d["scores"]["overextension"] >= sc.OX_OVERHEATED
    assert d["action"]["status"] in ("WAIT", "BUY_ON_PULLBACK")
    if sc.is_good_company(d["opp"]["score"], 10, 90):
        assert d["action"]["state"] == "GOOD_BUT_OVEREXTENDED"
    assert_consistent(d)


def test_no_ticker_specific_rules_in_decision_code():
    """決策程式裡不能有個股代號字串（docstring 的用法範例除外）。"""
    import ast
    import inspect
    from src.ranking import engine, scoring
    tickers = {"MRVL", "OKLO", "WDC", "STX", "AVGO", "LITE", "NRG", "VRT", "TER", "ARM", "INTC", "CEG", "VST", "BE", "CRDO", "RMBS"}
    for mod in (scoring, engine):
        tree = ast.parse(inspect.getsource(mod))
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
                docs.add(id(node.body[0].value))
        consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs}
        assert not (consts & tickers), f"{mod.__name__} 含個股特例 {consts & tickers}"
