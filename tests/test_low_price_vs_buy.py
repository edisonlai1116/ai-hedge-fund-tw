"""第二輪 Decision Logic Consistency：LOW PRICE ≠ BUY。

A. Price Location（是不是跌深）與 B. Investment Recommendation（值不值得投入新資金）分開；
Investment Quality Tier（排名 A/B/C/D）決定低檔規則可以多強；不改 Rank 本身。
迴歸案例用合成價格路徑重現使用者給的數字（Rank、距高點、6M、1M、波動），跑正式 features → decision_pipeline。
規則中沒有任何 ticker 判斷（test_no_ticker_specific_rules 檢查 scoring/engine/momentum/daily_advice）。
"""
import numpy as np
import pytest

from src.ranking import scoring as sc
from tests.test_decision_consistency import PRE_REVENUE, PROFITABLE, STRONG, assert_consistent, evaluate

BUY_RECS = ("BUY", "BUY_STAGED")


def _seg(n, total, vol, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    r = r - r.mean()
    out = np.cumprod(1 + (1 + total) ** (1 / n) - 1 + r)
    return out / out[-1] * (1 + total)          # 每段終點精準落在指定漲跌幅


def _path(segs):
    p = [100.0]
    for n, t, v, s in segs:
        p += list(p[-1] * _seg(n, t, v, s))
    return np.array(p[1:])


def _shape(c):
    return {"dd": c[-1] / c[-252:].max() - 1, "r6m": c[-1] / c[-127] - 1, "r1m": c[-1] / c[-22] - 1,
            "vol": float(np.std(np.diff(c[-64:]) / c[-64:-1]) * np.sqrt(252))}


OKLO = _path([(500, 3.0, 0.04, 1), (140, 4.0, 0.05, 2), (100, -0.70, 0.05, 3), (105, -0.12, 0.06, 4), (21, -0.15, 0.06, 5)])
NRG = _path([(600, 2.0, 0.02, 11), (30, 0.25, 0.025, 12), (21, -0.121, 0.03, 13), (105, -0.25, 0.03, 14), (21, -0.12, 0.03, 15)])
WDC = _path([(640, 2.5, 0.025, 21), (60, 1.22, 0.03, 22), (45, -0.365, 0.03, 23), (21, -0.15, 0.03, 24)])
ARM = _path([(650, 1.0, 0.03, 31), (50, 1.94, 0.04, 32), (55, -0.403, 0.04, 33), (21, 0.14, 0.04, 34)])
MRVL = _path([(640, 1.0, 0.04, 41), (60, 1.77, 0.015, 51), (45, -0.286, 0.015, 52), (21, 0.26, 0.02, 53)])
WEAK_PROFITABLE = {"operating_margin": 0.08, "fcf_margin": 0.03, "revenue": 2.8e10, "net_debt_to_ebitda": 7.2, "eps_revision_90d": -0.01}


# ---------------------------------------------------------------- 夾具忠於案例數字
@pytest.mark.parametrize("close,dd,r6m,r1m", [(OKLO, -0.79, -0.27, -0.15), (NRG, -0.42, -0.34, -0.12),
                                             (WDC, -0.46, 0.20, -0.15), (ARM, -0.32, 1.00, 0.14), (MRVL, -0.10, 1.49, 0.26)])
def test_fixtures_match_reported_numbers(close, dd, r6m, r1m):
    s = _shape(close)
    assert s["dd"] == pytest.approx(dd, abs=0.05)
    assert s["r6m"] == pytest.approx(r6m, abs=0.10)
    assert s["r1m"] == pytest.approx(r1m, abs=0.03)


# ---------------------------------------------------------------- 概念與規則（純函式）
def test_quality_tier_boundaries_follow_rank_not_drawdown():
    assert [sc.quality_tier(r, 102) for r in (1, 30, 31, 60, 61, 85, 86, 102)] == ["A", "A", "B", "B", "C", "C", "D", "D"]
    assert sc.quality_tier(16, 55) == "A" and sc.quality_tier(47, 55) == "D"      # 台股 55 檔依比例
    assert sc.quality_tier(percentile=95) == "A" and sc.quality_tier(percentile=5) == "D"


def test_price_location_is_only_location():
    assert sc.price_location(-0.79) == "DEEPLY_DISCOUNTED" and sc.price_location(-0.32) == "DEEPLY_DISCOUNTED"
    assert sc.price_location(-0.20) == "DISCOUNTED" and sc.price_location(-0.10) == "PULLBACK"
    assert sc.price_location(-0.02) == "NEAR_HIGH"
    assert sc.low_price_recommendation("PULLBACK", "A", "FUNDAMENTAL_DISCOUNT") is None      # 不是跌深就沒有低檔狀態


@pytest.mark.parametrize("dt", ["FUNDAMENTAL_DISCOUNT", "TEMPORARY_SHOCK", "VALUATION_RESET", "SPECULATIVE_DE_RATING",
                                "FUNDAMENTAL_DAMAGE", "UNKNOWN"])
def test_tier_table_low_price_never_buys_by_itself(dt):
    """Tier C 永遠只是觀察；Tier D 沒有獨立證據就不買；受損一律不買；投機下修沒有證據不買。"""
    rec = lambda t, conf=(): sc.low_price_recommendation("DEEPLY_DISCOUNTED", t, dt, list(conf))   # noqa: E731
    assert rec("C")["recommendation"] in ("WATCH", "NO_BUY", "SPECULATIVE_WATCH")
    assert rec("C")["label"] != "低檔布局可買（高排名回撤）"
    assert rec("D")["recommendation"] in ("SPECULATIVE_WATCH", "NO_BUY")
    assert rec("D")["recommendation"] not in BUY_RECS
    assert rec("D", ["CATALYST"])["recommendation"] in ("BUY_ON_CONFIRMATION", "NO_BUY", "SPECULATIVE_WATCH")   # 有證據最多是反轉候選
    if dt == "FUNDAMENTAL_DAMAGE":
        assert all(rec(t, ["CATALYST"])["recommendation"] == "NO_BUY" for t in "ABCD")
    if dt in sc.BUYABLE_DRAWDOWNS:
        assert rec("A")["recommendation"] == "BUY" and rec("A")["state"] == "LOW_PRICE_OPPORTUNITY"
        assert rec("B")["recommendation"] == "BUY_STAGED" and "中等排名" in rec("B")["label"]
    else:
        assert rec("A")["recommendation"] not in BUY_RECS and rec("B")["recommendation"] not in BUY_RECS


def test_three_distinct_low_price_states():
    assert sc.low_price_recommendation("DEEPLY_DISCOUNTED", "A", "FUNDAMENTAL_DISCOUNT")["state"] == "LOW_PRICE_OPPORTUNITY"
    assert sc.low_price_recommendation("DEEPLY_DISCOUNTED", "C", "FUNDAMENTAL_DISCOUNT")["state"] == "LOW_PRICE"
    assert sc.low_price_recommendation("DEEPLY_DISCOUNTED", "D", "FUNDAMENTAL_DISCOUNT")["state"] == "LOW_PRICE_SPECULATIVE"
    spec = sc.low_price_recommendation("DEEPLY_DISCOUNTED", "D", "UNKNOWN")
    assert "跌深本身不足以構成買進理由" in spec["why"]


def test_action_priority_damage_and_overextension_beat_location_and_rank():
    base = dict(opp=80, quality=80, risk=40, valuation=60, relative_strength=60, held=False, entry_ok=True, low_entry=True,
                growth=80, tier="A")
    assert sc.decide_action(overext=10, damage=80, drawdown_type="FUNDAMENTAL_DAMAGE", **base)["status"] == "WAIT"
    assert sc.decide_action(overext=90, drawdown_type="FUNDAMENTAL_DISCOUNT", **base)["status"] == "WAIT"
    assert sc.decide_action(overext=10, drawdown_type="FUNDAMENTAL_DISCOUNT", **{**base, "risk": 95})["status"] != "BUY_NOW"
    assert sc.decide_action(overext=10, drawdown_type="FUNDAMENTAL_DISCOUNT", **{**base, "entry_ok": False})["status"] != "BUY_NOW"
    assert sc.decide_action(overext=10, drawdown_type="FUNDAMENTAL_DISCOUNT", **{**base, "tier": "D"})["status"] == "WAIT"
    assert sc.decide_action(overext=10, drawdown_type="FUNDAMENTAL_DISCOUNT", **base)["status"] == "BUY_NOW"


# ---------------------------------------------------------------- 迴歸（正式流程）
def test_regression_oklo_rank96_down79_is_not_a_buy():
    """P5：Rank 96、-79%、6M -27%、1M -15%、波動 ~92% → DEEPLY_DISCOUNTED，但不是 BUY_NOW。"""
    fund = {"quality": 10, "growth": 60, "earnings_acceleration": 45, "valuation": 0, "ai_exposure": 70}
    for fm in (PRE_REVENUE, None):                       # 有基本面（投機下修）與沒有基本面（UNKNOWN）都不能買
        d = evaluate(OKLO, dict(fund), fund_metrics=fm, universe_rank=(96, 102))
        assert d["price_location"] == "DEEPLY_DISCOUNTED" and d["quality_tier"] == "D"
        assert d["action"]["status"] != "BUY_NOW"
        assert d["action"]["status"] == "WAIT"
        assert d["action"]["low_price"]["state"] == "LOW_PRICE_SPECULATIVE"
        assert d["action"]["low_price"]["recommendation"] == "SPECULATIVE_WATCH"
        assert_consistent(d)


def test_regression_nrg_rank101_down42_is_not_a_buy_unless_confirmed():
    """P6：Rank 101、-42%、6M -34%、1M -12% → DEEPLY_DISCOUNTED，不因跌深 BUY；有強烈獨立證據才是反轉候選（仍非 BUY_NOW）。"""
    weak = {"quality": 27, "growth": 23, "earnings_acceleration": 46, "valuation": 41, "ai_exposure": 45, "catalyst": 50}
    d = evaluate(NRG, dict(weak), fund_metrics=WEAK_PROFITABLE, universe_rank=(101, 102))
    assert d["price_location"] == "DEEPLY_DISCOUNTED" and d["quality_tier"] == "D"
    assert d["action"]["status"] == "WAIT"
    assert d["action"]["low_price"]["state"] == "LOW_PRICE_SPECULATIVE"
    assert_consistent(d)
    strong = {**weak, "earnings_acceleration": 80, "catalyst": 80}
    c = evaluate(NRG, dict(strong), fund_metrics={**WEAK_PROFITABLE, "eps_revision_90d": 0.02}, universe_rank=(101, 102))
    assert c["action"]["status"] != "BUY_NOW"
    if c["drawdown"]["type"] != "FUNDAMENTAL_DAMAGE":
        assert c["action"]["status"] == "BUY_ON_PULLBACK" and "REVERSAL_CANDIDATE" in c["action"]["flags"]
        assert c["action"]["low_price"]["recommendation"] == "BUY_ON_CONFIRMATION"
    assert_consistent(c)


def test_regression_wdc_rank59_down46_is_a_staged_low_price_opportunity():
    """P7：Rank 59（Tier B）、-46%、6M +20%、1M -15%、基本面完好 → 低檔機會（分批），理由是排名中等＋基本面未破壞。"""
    shock = {"shock": {"causes": ["macro", "industry"]}, "fundamental_damage_score": 10, "verdict": "POTENTIAL_OVERSOLD"}
    d = evaluate(WDC, {"quality": 85, "growth": 84, "earnings_acceleration": 85, "valuation": 50, "ai_exposure": 55},
                 shock_view=shock, fund_metrics=PROFITABLE, universe_rank=(59, 102))
    lp = d["action"]["low_price"]
    assert d["price_location"] == "DEEPLY_DISCOUNTED" and d["quality_tier"] == "B"
    assert lp["state"] == "LOW_PRICE_OPPORTUNITY" and lp["recommendation"] == "BUY_STAGED"
    assert "排名中等" in lp["why"] and "基本面未明顯破壞" in lp["why"]
    assert d["action"]["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")
    if d["action"]["status"] == "BUY_NOW":
        assert "STAGED_ENTRY" in d["action"]["flags"]
    assert_consistent(d)


def test_regression_arm_rank22_quality_pullback_still_buyable():
    """P8：Rank 22（Tier A）、-32%、6M +100%、1M +14% → BUY / BUY_ON_PULLBACK，不得降成 WAIT。"""
    d = evaluate(ARM, {"quality": 75, "growth": 80, "earnings_acceleration": 70, "valuation": 30, "ai_exposure": 85},
                 fund_metrics=PROFITABLE, universe_rank=(22, 102))
    assert d["quality_tier"] == "A" and d["price_location"] == "DEEPLY_DISCOUNTED"
    assert d["action"]["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")
    assert d["action"]["low_price"]["state"] == "LOW_PRICE_OPPORTUNITY"
    assert_consistent(d)


def test_regression_mrvl_rank14_hot_no_new_money():
    """P9：Rank 14、1M +26%、6M +149%、距高點 -10%、高波動 → 持有者續抱、新資金不買，理由是好公司＋強動能＋過熱。"""
    held = evaluate(MRVL, dict(STRONG), held=True, fund_metrics=PROFITABLE, universe_rank=(14, 102))
    new = held["new_money"]
    assert held["action"]["status"] in ("HOLD", "HOLD_CORE", "PARTIAL_PROFIT")
    if held["action"]["status"] == "HOLD":
        assert "過熱" in held["action"]["why"] and "不追" in held["action"]["why"]
    assert new["status"] in ("WAIT", "BUY_ON_PULLBACK") and new["status"] != "BUY_NOW"
    assert new["state"] == "GOOD_BUT_OVEREXTENDED"
    assert "過熱" in new["why"] and "不足" not in new["why"]
    assert held["price_location"] in ("PULLBACK", "NEAR_HIGH") and held["quality_tier"] == "A"
    assert_consistent(held)


def test_no_ticker_specific_rules_in_low_price_paths():
    """決策函式內不得出現個股代號（股票池清單是資料，不在檢查範圍）。"""
    import ast
    import inspect
    from src.strategy import daily_advice, momentum
    tickers = {"OKLO", "NRG", "MRVL", "ARM", "WDC", "STX", "LITE", "AVGO"}
    funcs = [momentum.low_view, momentum.evaluate_holdings, momentum.lookup_symbols, momentum.daily_holding_advice,
             momentum.score_row, momentum.rank_market, daily_advice._market_actions, daily_advice.build_daily_advice,
             sc.low_price_recommendation, sc.decide_action, sc.quality_tier, sc.price_location, sc.thesis_confirmations]
    for fn in funcs:
        tree = ast.parse(inspect.getsource(fn).lstrip())
        consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not (consts & tickers), f"{fn.__name__} 含個股特例 {consts & tickers}"


def test_strategy_path_low_view_uses_same_rule():
    """個股查詢／持股／每日建議（價格策略路徑）與機會評分用同一函式：Rank 96 跌深 → 不是「低檔布局可買」。"""
    from src.strategy.daily_advice import build_daily_advice
    from src.strategy.momentum import low_view
    rows = [
        {"symbol": "AAA", "name": "", "close": 36.0, "rank": 96, "dd_52w_pct": -79.0, "ret_3y_pct": 190.0, "high_52w": 170.0,
         "low_entry_price": 119.0, "low_entry": True, "low_entry_days": 1, "long_term_broken": False},
        {"symbol": "BBB", "name": "", "close": 400.0, "rank": 22, "dd_52w_pct": -33.0, "ret_3y_pct": 150.0, "high_52w": 600.0,
         "low_entry_price": 420.0, "low_entry": True, "low_entry_days": 1, "long_term_broken": False},
    ]
    report = {"markets": {"us": {"rows": rows, "low_entry": ["AAA", "BBB"], "low_entry_watch": [], "universe_size": 102},
                          "tw": {"rows": [], "low_entry": [], "low_entry_watch": [], "universe_size": 55}},
              "drawdown_types": {"AAA": {"type": "FUNDAMENTAL_DISCOUNT"}, "BBB": {"type": "FUNDAMENTAL_DISCOUNT"}},
              "sentiment": {"fear_greed": {"score": 50, "label": "中性"}, "vix": {}, "sectors": []}}
    assert low_view(rows[0], report)["recommendation"] == "SPECULATIVE_WATCH"
    assert low_view(rows[1], report)["recommendation"] == "BUY"
    adv = build_daily_advice(report)["markets"]["us"]
    buys = {a["symbol"] for a in adv["actions"] if a["type"] == "low_entry_new"}
    assert buys == {"BBB"}
    aaa = next(w for w in adv["watch"] if w["symbol"] == "AAA")
    assert "可買" not in aaa["action"] and aaa["low_state"] == "LOW_PRICE_SPECULATIVE"
