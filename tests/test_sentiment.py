"""市場情緒：自建恐懼貪婪 point-in-time、情緒規則不偷看未來、每日建議輸出。"""
import numpy as np
import pandas as pd

from src.strategy import sentiment as sm

DAYS = pd.bdate_range("2015-01-01", periods=900)


def _closes(seed=0, tweak_after=None):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ["SPY", "TLT", "IEF", "HYG", "^VIX", "^VIX3M"] + list(sm.SECTOR_ETFS):
        base = 20.0 if k.startswith("^") else 100.0
        s = pd.Series(base * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS)))), index=DAYS)
        if tweak_after is not None:
            s.iloc[tweak_after:] *= 2.0
        out[k] = s
    return out


def test_fear_greed_proxy_is_point_in_time():
    t = 700
    a = sm.fear_greed_proxy(_closes())["fg"]
    b = sm.fear_greed_proxy(_closes(tweak_after=t))["fg"]
    pd.testing.assert_series_equal(a.iloc[:t], b.iloc[:t])
    assert a.dropna().between(0, 100).all()


def test_fg_label_buckets():
    assert sm.fg_label(10) == "極度恐懼"
    assert sm.fg_label(50) == "中性"
    assert sm.fg_label(90) == "極度貪婪"


def _universe(seed=1, n=30):
    rng = np.random.default_rng(seed)
    cl = pd.DataFrame({f"S{i}": 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(DAYS)))) for i in range(n)}, index=DAYS)
    return cl, cl.shift(1).bfill() * (1 + rng.normal(0, 0.002, cl.shape))


def test_panic_rule_and_idle_do_not_look_ahead():
    """只改 T 日以後的價格與恐懼貪婪 → T 日以前的權益曲線不變。"""
    from src.strategy import backtest as sb
    cl, op = _universe()
    fg = pd.Series(np.where(np.arange(len(DAYS)) % 50 < 10, 10.0, 60.0), index=DAYS)
    t = 820
    start = str(DAYS[400].date())

    def run(cl_, op_, fg_):
        r = sb.run_backtest("us", cl_, cl_["S0"], start=start, opens=op_, fg=fg_)
        e = r["_eq_daily"]
        low, _ = sb.lowentry_returns(cl_, start=start, opens=op_, idle_ret=e.pct_change().shift(1))
        return e, low

    e1, l1 = run(cl, op, fg)
    cl2, op2, fg2 = cl.copy(), op.copy(), fg.copy()
    cl2.iloc[t:] *= 2.0
    op2.iloc[t:] *= 2.0
    fg2.iloc[t:] = 0.0
    e2, l2 = run(cl2, op2, fg2)
    cut = DAYS[t - 2]
    pd.testing.assert_series_equal(e1[:cut], e2[:cut])
    pd.testing.assert_series_equal(l1[:cut], l2[:cut])


def test_panic_rule_keeps_holdings_at_rebalance():
    from src.strategy import backtest as sb
    cl, op = _universe()
    start = str(DAYS[400].date())
    calm = sb.run_backtest("us", cl, cl["S0"], start=start, opens=op, fg=pd.Series(60.0, index=DAYS))
    panic = sb.run_backtest("us", cl, cl["S0"], start=start, opens=op, fg=pd.Series(10.0, index=DAYS))
    sells = lambda r: sum(len(x["sell"]) for x in r["recent_rebalances"])
    assert sells(panic) == 0
    assert sells(calm) >= sells(panic)


def test_daily_advice_levels():
    from src.strategy.daily_advice import build_daily_advice
    row = {"symbol": "AAA", "name": "", "close": 70.0, "rank": 5, "dd_52w_pct": -31.0, "ret_3y_pct": 50.0,
           "high_52w": 101.0, "low_entry_price": 70.7, "low_entry": True, "low_entry_days": 1}
    report = {
        "sentiment": {"fear_greed": {"score": 20.0, "label": "極度恐懼"}, "vix": {"value": 32.0}, "sectors": []},
        "markets": {"us": {"rows": [row], "low_entry": ["AAA"], "low_entry_watch": [], "momentum_share": 0.93},
                    "tw": {"rows": [], "low_entry": [], "low_entry_watch": []}},
    }
    # 沒有回撤分類（或分類為投機修正）→ 跌深不等於便宜，不能直接給「買進」
    adv = build_daily_advice(report)
    assert not any(a["type"] == "low_entry_new" for a in adv["markets"]["us"]["actions"])
    assert adv["markets"]["us"]["watch"][0]["type"] == "low_entry_not_buyable"
    report["drawdown_types"] = {"AAA": {"type": "SPECULATIVE_DE_RATING", "why": "x"}}
    assert not any(a["type"] == "low_entry_new" for a in build_daily_advice(report)["markets"]["us"]["actions"])
    report["drawdown_types"] = {"AAA": {"type": "FUNDAMENTAL_DISCOUNT", "why": "x"}}
    adv = build_daily_advice(report)
    assert adv["panic_no_sell"] is True
    assert adv["markets"]["us"]["level"] == "action"
    assert adv["markets"]["us"]["actions"][0]["type"] == "low_entry_new"
    assert adv["markets"]["tw"]["level"] in ("hold", "action")   # 月調窗口時為 action
    assert any("極度恐懼" in g for g in adv["sentiment_guidance"])


def test_tech_scores_do_not_look_ahead():
    """virattt 技術分數：只改月調訊號日之後的 K 線 → 該日分數不變；混入排名後回測 T 以前權益不變。"""
    from src.strategy import backtest as sb
    cl, op = _universe(n=12)
    pm = {k: pd.DataFrame({"Open": op[k], "High": cl[k] * 1.01, "Low": cl[k] * 0.99, "Close": cl[k], "Volume": 1e6}) for k in cl}
    start = str(DAYS[400].date())
    t = 820
    pm2 = {k: f.copy() for k, f in pm.items()}
    for f in pm2.values():
        f.iloc[t:] *= 2.0
    a = sb.tech_scores_at_rebalances(pm, cl.index, start=start)
    b = sb.tech_scores_at_rebalances(pm2, cl.index, start=start)
    for d in a:
        if d < DAYS[t]:
            pd.testing.assert_series_equal(a[d], b[d])
    assert any(len(v) for v in a.values())
    cl2, op2 = cl.copy(), op.copy()
    cl2.iloc[t:] *= 2.0
    op2.iloc[t:] *= 2.0
    e1 = sb.run_backtest("tw", cl, cl["S0"], start=start, opens=op, tech=a)["_eq_daily"]
    e2 = sb.run_backtest("tw", cl2, cl2["S0"], start=start, opens=op2, tech=b)["_eq_daily"]
    pd.testing.assert_series_equal(e1[:DAYS[t - 2]], e2[:DAYS[t - 2]])


def test_no_chase_rules_do_not_look_ahead_and_wait_for_pullback():
    """不追大長紅：限價單/等待只用 T-1 以前資料；大長紅後不在隔天開盤買。"""
    from src.strategy import backtest as sb
    cl, op = _universe(seed=3)
    rng = np.random.default_rng(7)
    for k in cl.columns[:10]:                       # 製造大長紅
        for t0 in rng.integers(450, 850, 6):
            cl.iloc[t0:, cl.columns.get_loc(k)] *= 1.15
            op.iloc[t0 + 1:, op.columns.get_loc(k)] *= 1.15
    lo = np.minimum(cl, op) * 0.98
    start = str(DAYS[400].date())
    t = 820

    def run(cl_, op_, lo_):
        r = sb.run_backtest("us", cl_, cl_["S0"], start=start, opens=op_)["_eq_daily"]
        l, _ = sb.lowentry_returns(cl_, start=start, opens=op_, idle_ret=r.pct_change().shift(1), lows=lo_)
        return r, l

    r1, l1 = run(cl, op, lo)
    cl2, op2, lo2 = cl.copy(), op.copy(), lo.copy()
    for x in (cl2, op2, lo2):
        x.iloc[t:] *= 2.0
    r2, l2 = run(cl2, op2, lo2)
    cut = DAYS[t - 2]
    pd.testing.assert_series_equal(r1[:cut], r2[:cut])
    pd.testing.assert_series_equal(l1[:cut], l2[:cut])
