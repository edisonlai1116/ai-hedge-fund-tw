"""Look-ahead 擾動測試：只改 T 日收盤（Case A=100、Case B=200），T 日（含）以前的決策、成交與權益
必須完全相同。若不同，代表有人拿 T 日收盤去做 T 日或更早的交易（look-ahead bias）。"""
import numpy as np
import pandas as pd
import pytest

DAYS = pd.bdate_range("2024-01-01", periods=60)
T = DAYS[30]


def _ohlcv(close_at_t: float) -> pd.DataFrame:
    base = np.linspace(100, 130, len(DAYS))
    o = base.copy()
    c = base + 0.5
    c[30] = close_at_t                 # 只改 T 日收盤；T+1 開盤與之後全部相同
    return pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) + 1, "low": np.minimum(o, c) - 1,
                         "volume": 1000}, index=pd.DatetimeIndex(DAYS, name="Date"))


# ---------------- 1) LLM/agent 回測引擎 ----------------
def test_engine_trades_up_to_T_do_not_depend_on_T_close(monkeypatch):
    pytest.importorskip("langchain_core")
    from src.backtesting import engine as eng_mod
    from src.backtesting.engine import BacktestEngine

    def run(close_at_t):
        bars = {"AAA": _ohlcv(close_at_t)}

        def price_data(ticker, start, end, api_key=None):
            df = bars[ticker]
            return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

        monkeypatch.setattr(eng_mod, "get_price_data", price_data)
        for f in ("get_prices", "get_financial_metrics", "get_insider_trades", "get_company_news"):
            monkeypatch.setattr(eng_mod, f, lambda *a, **k: [])
        monkeypatch.setattr(eng_mod.BenchmarkCalculator, "get_return_pct", lambda *a, **k: 0.0)
        log = []

        def agent(*, tickers, start_date, end_date, **kw):
            # 依「看得到的最新收盤」交易：收盤 > 前一日 → 買 1 股，否則賣 1 股
            df = price_data("AAA", start_date, end_date)
            act = "buy" if len(df) > 1 and df["close"].iloc[-1] > df["close"].iloc[-2] else "sell"
            log.append((end_date, act))
            return {"decisions": {"AAA": {"action": act, "quantity": 1}}, "analyst_signals": {}}

        e = BacktestEngine(agent=agent, tickers=["AAA"], start_date=str(DAYS[0].date()), end_date=str(DAYS[-1].date()),
                           initial_capital=10000.0, model_name="m", model_provider="p", selected_analysts=None,
                           initial_margin_requirement=0.0)
        e._results.print_rows = lambda rows: None
        e.run_backtest()
        vals = {pd.Timestamp(v["Date"]): v["Portfolio Value"] for v in e.get_portfolio_values()}
        return log, vals

    log_a, vals_a = run(100.0)
    log_b, vals_b = run(200.0)
    # 決策：end_date < T 的決策完全相同（T 日收盤只能影響 T+1 開盤的交易）
    assert [x for x in log_a if pd.Timestamp(x[0]) < T] == [x for x in log_b if pd.Timestamp(x[0]) < T]
    # T 日以前（不含 T 的收盤估值）權益相同
    assert {d: v for d, v in vals_a.items() if d < T} == {d: v for d, v in vals_b.items() if d < T}
    # T 日收盤後的決策（在 T+1 開盤成交）才可以不同
    assert dict(log_a)[str(T.date())] != dict(log_b)[str(T.date())]


# ---------------- 2) Ranking 回測（open-to-open） ----------------
def _panel(close_at_t):
    df = _ohlcv(close_at_t)
    close = pd.DataFrame({"A": df["close"], "B": 230 - df["close"]})
    opens = pd.DataFrame({"A": df["open"], "B": 230 - df["open"]})
    return close, opens


def test_ranking_backtest_equity_up_to_T_unchanged():
    from src.ranking.backtest import run_strategy
    out = {}
    for case, c in (("A", 100.0), ("B", 200.0)):
        close, opens = _panel(c)
        score = close.pct_change(5)                 # 分數只用收盤
        out[case] = run_strategy(score, opens, top=1, keep=1, reb=1, start=str(DAYS[10].date()))["equity"]
    pd.testing.assert_series_equal(out["A"].loc[:T], out["B"].loc[:T])


# ---------------- 3) 網站動能回測（src/strategy/backtest.run_backtest） ----------------
def test_website_momentum_backtest_up_to_T_unchanged(monkeypatch):
    import src.strategy.backtest as sb
    monkeypatch.setattr(sb, "LOOKBACK", 3)
    monkeypatch.setattr(sb, "REBALANCE_DAYS", 1)
    monkeypatch.setitem(sb.TOP_N, "us", 1)
    monkeypatch.setitem(sb.KEEP_N, "us", 1)
    out = {}
    for case, c in (("A", 100.0), ("B", 200.0)):
        close, opens = _panel(c)
        res = sb.run_backtest("us", close, close["A"], start=str(DAYS[5].date()), opens=opens)
        out[case] = res["_eq_daily"]
    pd.testing.assert_series_equal(out["A"].loc[:T], out["B"].loc[:T])


# ---------------- 4) 長線低檔布局回測 ----------------
def test_lowentry_never_buys_at_signal_close():
    from src.strategy.backtest import lowentry_returns
    days = pd.bdate_range("2019-01-01", periods=1100)
    t = 1000
    out = {}
    for case, c_t in (("A", 100.0), ("B", 60.0)):          # B：T 收盤跌 40% → 觸發低檔訊號
        px = np.r_[np.linspace(50, 100, 900), np.full(200, 100.0)]
        px[t] = c_t
        if case == "B":
            px[t + 1:] = 60.0
        closes = pd.DataFrame({"X": px}, index=days)
        out[case] = lowentry_returns(closes, start=str(days[800].date()))[0]
    T_ = days[t]
    # T 日與 T+1 日：沒有任何持股能賺到 T 收盤 → T+1 的報酬（訊號後最早 T+1 收盤才進場）
    assert out["A"].loc[:days[t + 1]].abs().sum() == 0
    assert out["B"].loc[:days[t + 1]].abs().sum() < 1e-9 + 0.002 * 2   # 只有進場成本
    assert out["B"].loc[T_] == 0.0


# ---------------- 5) 舊版個股趨勢回測（simple_signal.compute_timeline_backtest） ----------------
def test_legacy_timeline_backtest_enters_after_signal_bar():
    from src.simple_signal import compute_timeline_backtest, compute_macd, compute_rsi
    days = pd.bdate_range("2023-01-01", periods=200)
    c = pd.Series(np.r_[np.linspace(100, 90, 150), np.linspace(90, 120, 50)], index=days)
    f = pd.DataFrame({"Close": c, "Volume": 1e6})
    f["MA20"], f["MA50"], f["MA120"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(120).mean()
    f["RSI14"] = compute_rsi(c, 14)
    m, s_, _ = compute_macd(c)
    f["MACD"], f["MACD_Signal"] = m, s_
    f = f.dropna()
    res = compute_timeline_backtest("TEST", f)
    sig = f.index[(f["MACD"] > f["MACD_Signal"]) & (f["MACD"].shift() <= f["MACD_Signal"].shift())]
    for tr in res["trades_log"]:
        assert pd.Timestamp(tr["entry_date"]) not in set(sig)   # 永遠不在訊號當日收盤進場
