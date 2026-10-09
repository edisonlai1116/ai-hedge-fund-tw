"""持股決策層的情境測試（離線、不連網；基本面用合成的機會評分列）。"""
import pandas as pd

from src.strategy import holding_decision as hd
from src.strategy import momentum as m


def _report(n=150, overrides=None):
    rows = []
    for i in range(1, n + 1):
        sym = f"S{i:03d}"
        r = {"symbol": sym, "name": "", "rank": i, "zone": m.zone_of(i, "us"), "score": 5 - i * 0.01,
             "close": 100.0, "ret_6m_pct": 10.0, "ret_3y_pct": 50.0, "dd_52w_pct": -5.0, "high_52w": 105.0,
             "low_entry": False, "long_term_broken": False, "low_entry_price": 73.5, "quality_tier": "A" if i <= 45 else "C"}
        r.update((overrides or {}).get(sym, {}))
        rows.append(r)
    return {"generated_at": "2026-10-10T00:00:00+08:00",
            "strategy": {"next_rebalance": "2026-11-02", "in_rebalance_window": False},
            "markets": {"us": {"universe_size": n, "rows": rows, "low_entry": [r["symbol"] for r in rows if r["low_entry"]]},
                        "tw": {"universe_size": 0, "rows": []}}}


def _fund(q, g, acc, val, dd_type, conf=(), damage=None, speculative=False, ok_now=True):
    return {"scores": {"quality": q, "growth": g, "earnings_acceleration": acc, "valuation": val},
            "drawdown_type": dd_type, "drawdown": {"type": dd_type, "why": "test"}, "confirmations": list(conf),
            "shock": {"fundamental_damage_score": damage} if damage is not None else None,
            "risk_detail": {"fundamental_uncertainty": {"speculative": speculative, "evidence": []}},
            "downside_risk": 0.2, "vol_63": 0.45, "valuation_class": "A",
            "entry": {"zone_1": {"low": 74, "high": 76, "basis": "20 日低點"}, "zone_2": {"low": 70, "high": 72, "basis": "1 ATR"},
                      "zone_3": {"low": 65, "high": 67, "basis": "2 ATR"}, "atr": 3.0, "condition": {"ok": ok_now}}}


VOO = {"VOO": pd.DataFrame({"Close": [500.0] * 10})}


def _eval(holdings, rep, fund, cash=0.0, decisions=True):
    return m.evaluate_holdings(holdings, rep, fx_usd_twd=30.0, extra_prices=dict(VOO), cash_twd=cash,
                               decisions=decisions, fundamentals=fund, live=False)


def _row(ev, sym):
    return next(h for h in ev["holdings"] if h["symbol"] == sym)


def test_quality_stock_in_market_correction_is_not_sold_and_can_be_scaled_into():
    # 第 60 名（跌出保留名單 50）、從高點跌 25%、帳面虧損 15%，但基本面強、下跌屬暫時衝擊、已止跌反轉
    rep = _report(overrides={"S060": {"dd_52w_pct": -25.0, "close": 75.0}})
    fund = {"S060": _fund(78, 70, 65, 68, "TEMPORARY_SHOCK", conf=["POSITIVE_REVERSAL"])}
    holds = [{"ticker": "S060", "cost": 88.2, "shares": 50}, {"ticker": "VOO", "cost": 400, "shares": 400}]
    base = _eval(holds, rep, fund, decisions=False)
    assert _row(base, "S060")["action"] == "賣出換股"                       # 原規則：排名跌出就賣
    ev = _eval(holds, rep, fund)
    h = _row(ev, "S060")
    assert h["decision"]["group"] == "LEFT_SIDE"
    assert h["decision"]["add_grade"] in ("左側布局可買", "低檔分批")
    assert h["action"] != "賣出換股" and not str(h["today"]).startswith(("下一步換股", "今天賣出"))
    plan = h["decision"]["plan"]
    assert len(plan["tranches"]) == 3 and plan["invalidation"]
    assert abs(sum(t["pct"] for t in plan["tranches"]) - 100) <= 1


def test_loss_stock_with_fundamental_damage_is_not_averaged_down():
    # 低檔區（-40%）、帳面虧損 35%、排名 30，但回撤分類＝基本面受損
    rep = _report(overrides={"S030": {"dd_52w_pct": -40.0, "close": 60.0, "low_entry": True}})
    fund = {"S030": _fund(40, 30, 20, 45, "FUNDAMENTAL_DAMAGE", damage=80)}
    ev = _eval([{"ticker": "S030", "cost": 92, "shares": 20}, {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    h = _row(ev, "S030")
    assert h["decision"]["add_grade"] == "不宜加碼"
    assert h["decision"]["thesis"] == "broken"
    assert h["decision"]["group"] == "EXIT" and h["decision"]["decision"] == "EXIT_ALL"
    assert h["action"] not in ("低檔加碼", "加碼")


def test_loss_alone_is_not_a_sell_reason_and_profit_alone_is_not_a_hold_reason():
    rep = _report()
    fund = {"S005": _fund(80, 75, 60, 55, "NONE"), "S070": _fund(30, 25, 20, 30, "NONE")}
    ev = _eval([{"ticker": "S005", "cost": 150, "shares": 10},     # 虧損 33%，但排名第 5、基本面強
                {"ticker": "S070", "cost": 20, "shares": 10},      # 獲利 400%，但排名 70、品質成長都低
                {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    assert _row(ev, "S005")["decision"]["group"] == "CORE"
    assert _row(ev, "S070")["decision"]["group"] == "EXIT"


def test_wait_group_lists_missing_evidence_and_conditions():
    rep = _report(overrides={"S020": {"dd_52w_pct": -30.0, "close": 70.0}})
    fund = {"S020": _fund(60, 55, 45, 50, "UNKNOWN")}
    ev = _eval([{"ticker": "S020", "cost": 90, "shares": 20}, {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    d = _row(ev, "S020")["decision"]
    assert d["group"] == "WAIT"
    w = d["wait"]
    assert w["missing_evidence"] and w["start_condition"] and w["exit_condition"] and w["watch"]


def test_no_fundamentals_marks_data_insufficient_and_keeps_baseline_rule():
    rep = _report(overrides={"S090": {"dd_52w_pct": -30.0}})
    ev = _eval([{"ticker": "S090", "cost": 120, "shares": 10}, {"ticker": "VOO", "cost": 400, "shares": 400}], rep, {})
    d = _row(ev, "S090")["decision"]
    assert d["add_grade"] == "資料不足，不能判定"
    assert d["thesis"] == "unknown"
    assert "基本面資料（抓不到）" in d["confidence"]["missing"]
    assert _row(ev, "S090")["action"] == "賣出換股"                         # 資料不足 → 沿用回測規則


def test_multiple_cost_lots_are_merged_into_one_holding():
    rep = _report()
    ev = _eval([{"ticker": "S003", "cost": 80, "shares": 10}, {"ticker": "S003", "cost": 120, "shares": 30},
                {"ticker": "VOO", "cost": 400, "shares": 400}], rep, {"S003": _fund(70, 70, 60, 50, "NONE")})
    rows = [h for h in ev["holdings"] if h["symbol"] == "S003"]
    assert len(rows) == 1
    assert rows[0]["shares"] == 40 and abs(rows[0]["cost"] - 110) < 1e-9 and rows[0]["lots"] == 2


def test_cash_scenarios_sell_exit_candidates_first_and_report_portfolio_after():
    rep = _report(overrides={"S030": {"dd_52w_pct": -40.0, "close": 60.0}})
    fund = {"S030": _fund(40, 30, 20, 45, "FUNDAMENTAL_DAMAGE", damage=80), "S004": _fund(80, 80, 70, 60, "NONE")}
    ev = _eval([{"ticker": "S030", "cost": 90, "shares": 300}, {"ticker": "S004", "cost": 80, "shares": 300},
                {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    total = ev["total_twd"]
    sc = hd.cash_scenarios(ev["holdings"], total, 0.0, 30.0, targets=(300_000, 1_000_000))
    assert sc[0]["sells"][0]["symbol"] == "S030"                         # 退出候選先賣
    assert sc[0]["enough"] and abs(sc[0]["raised_twd"] - 300_000) < 1
    assert sc[1]["after"]["cash_pct"] > sc[0]["after"]["cash_pct"]


def test_report_is_consistent_between_summary_groups_and_lists():
    rep = _report(overrides={"S060": {"dd_52w_pct": -25.0, "close": 75.0},
                             "S030": {"dd_52w_pct": -40.0, "close": 60.0}})
    fund = {"S060": _fund(78, 70, 65, 68, "TEMPORARY_SHOCK", conf=["POSITIVE_REVERSAL"]),
            "S030": _fund(40, 30, 20, 45, "FUNDAMENTAL_DAMAGE", damage=80), "S002": _fund(80, 80, 70, 60, "NONE")}
    holds = [{"ticker": "S060", "cost": 88, "shares": 50}, {"ticker": "S030", "cost": 90, "shares": 50},
             {"ticker": "S002", "cost": 50, "shares": 50}, {"ticker": "VOO", "cost": 400, "shares": 400}]
    ev = _eval(holds, rep, fund, cash=200_000)
    base = _eval(holds, rep, fund, cash=200_000, decisions=False)
    r = hd.build_decision_report(ev, None, 200_000, 30.0, baseline=base)
    left = {x["symbol"] for x in r["left_side"]}
    assert left == set(r["groups"]["左側布局候選"])
    assert {x["symbol"] for x in r["sells"]} >= set(r["groups"]["退出候選"]) - left
    for h in ev["holdings"]:
        g = h["decision"]["group"]
        if g == "EXIT":
            assert h["action"] not in ("低檔加碼", "加碼")                      # 退出候選不會同時建議加碼
        if str(h.get("today", "")).startswith("今天"):
            assert h["symbol"] in r["overall"]["verdict"]                     # 個股今天動作一定出現在每日總結
    assert any(x["symbol"] == "S060" for x in r["baseline_diff"])             # 前後對照列出被改動的持股
    assert r["data_confidence"]["S060"]["missing"]


def test_concentration_is_partial_trim_not_full_exit():
    rep = _report()
    fund = {"S001": _fund(85, 85, 70, 55, "NONE")}
    ev = _eval([{"ticker": "S001", "cost": 50, "shares": 3000}, {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    d = _row(ev, "S001")["decision"]
    assert d["decision"] == "TRIM" and d["group"] == "CORE"
    assert _row(ev, "S001")["trim_shares"] < 3000


def test_missing_scores_do_not_crash_and_report_still_builds():
    # 線上真的發生過：有基本面列、但品質／估值等分數是 None → 不可讓整個持股評估 500
    rep = _report(overrides={"S060": {"dd_52w_pct": -25.0, "close": 75.0}})
    fund = {"S060": _fund(None, 70, None, None, "TEMPORARY_SHOCK"), "S002": _fund(None, None, None, None, "UNKNOWN")}
    ev = _eval([{"ticker": "S060", "cost": 88, "shares": 50}, {"ticker": "S002", "cost": 50, "shares": 10},
                {"ticker": "VOO", "cost": 400, "shares": 400}], rep, fund)
    assert all(h.get("decision") for h in ev["holdings"])
    assert _row(ev, "S002")["decision"]["thesis"] == "unknown"
    r = hd.build_decision_report(ev, None, 0.0, 30.0)
    assert r["overall"]["verdict"]
