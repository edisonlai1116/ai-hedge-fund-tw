"""Point-in-time 回測保證：T 日的決策不可能讀到 T 日收盤；成交一律用 T 日開盤價。"""
import datetime

import pandas as pd
import pytest

from src.backtesting import engine as engine_mod
from src.backtesting.engine import EXECUTION_TIMING, BacktestEngine
from src.tools.api import _filter_point_in_time, pit_report_cutoff

TICKERS = ["AAA", "BBB"]
DAYS = pd.bdate_range("2024-03-01", "2024-03-15")


def _bars(ticker: str) -> pd.DataFrame:
    base = 100.0 if ticker == "AAA" else 50.0
    rows = []
    for i, d in enumerate(DAYS):
        o = base + i                 # 開盤
        rows.append({"open": o, "close": o + 0.5, "high": o + 1, "low": o - 1, "volume": 1000})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(DAYS, name="Date"))


BARS = {t: _bars(t) for t in TICKERS}


def _fake_price_data(ticker, start_date, end_date, api_key=None):
    df = BARS[ticker]
    return df.loc[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]


class SpyAgent:
    """記錄每次被呼叫時的 end_date，並只能透過「截到 end_date」的資料做決策。"""

    def __init__(self):
        self.calls = []

    def __call__(self, *, tickers, start_date, end_date, **kwargs):
        visible = {t: _fake_price_data(t, start_date, end_date) for t in tickers}
        latest_seen = max(df.index.max() for df in visible.values())
        self.calls.append({"end_date": end_date, "start_date": start_date, "latest_seen": latest_seen})
        decisions = {t: {"action": "hold", "quantity": 0} for t in tickers}
        if len(self.calls) == 1:
            decisions["AAA"] = {"action": "buy", "quantity": 10}
        return {"decisions": decisions, "analyst_signals": {}}


@pytest.fixture
def run_engine(monkeypatch):
    monkeypatch.setattr(engine_mod, "get_price_data", _fake_price_data)
    monkeypatch.setattr(engine_mod, "get_prices", lambda *a, **k: None)
    monkeypatch.setattr(engine_mod, "get_financial_metrics", lambda *a, **k: [])
    monkeypatch.setattr(engine_mod, "get_insider_trades", lambda *a, **k: [])
    monkeypatch.setattr(engine_mod, "get_company_news", lambda *a, **k: [])
    monkeypatch.setattr(engine_mod.BenchmarkCalculator, "get_return_pct", lambda *a, **k: 0.0)

    def _run():
        agent = SpyAgent()
        eng = BacktestEngine(
            agent=agent, tickers=TICKERS, start_date="2024-03-01", end_date="2024-03-15",
            initial_capital=100000.0, model_name="m", model_provider="p",
            selected_analysts=None, initial_margin_requirement=0.5,
        )
        eng._results.print_rows = lambda rows: None
        metrics = eng.run_backtest()
        return eng, agent, metrics

    return _run


def test_decision_on_day_t_never_sees_day_t_data(run_engine):
    eng, agent, _ = run_engine()
    trading_days = [d.strftime("%Y-%m-%d") for d in DAYS]
    assert agent.calls, "agent 應該被呼叫"
    for call in agent.calls:
        exec_day = trading_days[trading_days.index(call["end_date"]) + 1]
        # 決策的資料截止日一定早於成交日，且看到的最新 K 線就是 end_date（T-1）。
        assert call["end_date"] < exec_day
        assert call["latest_seen"] == pd.Timestamp(call["end_date"])
        assert call["latest_seen"] < pd.Timestamp(exec_day)


def test_first_day_has_no_decision(run_engine):
    _, agent, _ = run_engine()
    assert agent.calls[0]["end_date"] == "2024-03-01"   # 3/1 收盤後的訊號 → 3/4 開盤成交
    assert len(agent.calls) == len(DAYS) - 1


def test_execution_uses_open_not_close(run_engine):
    eng, agent, _ = run_engine()
    exec_day = pd.Timestamp("2024-03-04")              # 第一個決策（3/1 收盤後）在 3/4 成交
    expected_open = float(BARS["AAA"].loc[exec_day, "open"])
    expected_close = float(BARS["AAA"].loc[exec_day, "close"])
    pos = eng._portfolio.get_snapshot()["positions"]["AAA"]
    assert pos["long"] == 10
    assert pos["long_cost_basis"] == pytest.approx(expected_open)
    assert pos["long_cost_basis"] != pytest.approx(expected_close)


def test_lookback_is_at_least_twelve_months(run_engine):
    _, agent, _ = run_engine()
    for call in agent.calls:
        span = pd.Timestamp(call["end_date"]) - pd.Timestamp(call["start_date"])
        assert span.days >= 365


def test_execution_timing_is_reported(run_engine):
    eng, _, metrics = run_engine()
    assert metrics["execution_timing"] == EXECUTION_TIMING
    assert "T open" in EXECUTION_TIMING and "T-1 close" in EXECUTION_TIMING


def test_financials_respect_publication_lag():
    today = datetime.date(2026, 10, 6)
    # 回測日 2024-05-10：Q1（3/31 期末）要到 ~5/15 才公布 → 只能用 2024-03-26 以前期末的財報
    assert pit_report_cutoff("2024-05-10", "quarterly", today=today) == "2024-03-26"
    assert pit_report_cutoff("2024-05-10", "annual", today=today) == "2024-02-10"
    # 即時（今天）不套延遲
    assert pit_report_cutoff("2026-10-06", "ttm", today=today) == "2026-10-06"

    class R:
        def __init__(self, rp):
            self.report_period = rp

    kept = _filter_point_in_time([R("2024-03-31"), R("2023-12-31")], "2024-03-26")
    assert [k.report_period for k in kept] == ["2023-12-31"]
