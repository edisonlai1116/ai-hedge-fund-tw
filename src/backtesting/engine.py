from __future__ import annotations

from datetime import datetime
from typing import Sequence, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta

from .controller import AgentController
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .portfolio import Portfolio
from .types import PerformanceMetrics, PortfolioValuePoint
from .valuation import calculate_portfolio_value, compute_exposures
from .output import OutputBuilder
from .benchmarks import BenchmarkCalculator

from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)


# 回測時序（point-in-time）：T-1 收盤後產生訊號 → T 日開盤成交 → T 日收盤估值。
# 2026-10-06 修正 look-ahead bias：舊版在 T 日用 end_date=T（含 T 日收盤）產生決策、
# 又用 T 日收盤價成交，等於「看完收盤價再用收盤價下單」。
EXECUTION_TIMING = "signal: T-1 close (data <= T-1) -> execution: T open -> mark-to-market: T close"

# 決策可用歷史：13 個月（>= 252 個交易日），讓 EMA55 / MA50 / 126 日動能 / 63 日波動 /
# 布林通道 / ATR 都有足夠資料；舊版只給 1 個月，長天期指標全是 NaN 卻仍被拿來投票。
LOOKBACK_MONTHS = 13


class BacktestEngine:
    """Coordinates the backtest loop using the new components.

    Point-in-time contract（見 EXECUTION_TIMING）：
      * 交易日 T 的決策只呼叫 agent(end_date=前一個交易日)，任何 agent 都拿不到 T 日資料。
      * 成交價一律用 T 日開盤價（open），不用收盤價。
      * 估值（portfolio value）用 T 日收盤價，只做 mark-to-market，不影響決策。
    """

    def __init__(
        self,
        *,
        agent,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        initial_margin_requirement: float,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts

        self._portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_capital,
            margin_requirement=initial_margin_requirement,
        )
        self._executor = TradeExecutor()
        self._agent_controller = AgentController()
        self._perf = PerformanceMetricsCalculator()
        self._results = OutputBuilder(initial_capital=self._initial_capital)

        # Benchmark calculator
        self._benchmark = BenchmarkCalculator()

        self._portfolio_values: list[PortfolioValuePoint] = []
        self._table_rows: list[list] = []
        self._performance_metrics: PerformanceMetrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "long_short_ratio": None,
            "gross_exposure": None,
            "net_exposure": None,
        }

    @property
    def execution_timing(self) -> str:
        return EXECUTION_TIMING

    def _prefetch_data(self) -> None:
        start_date_dt = datetime.strptime(self._start_date, "%Y-%m-%d")
        start_date_str = (start_date_dt - relativedelta(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")

        for ticker in self._tickers:
            get_prices(ticker, start_date_str, self._end_date)
            get_financial_metrics(ticker, self._end_date, limit=10)
            insider_start = (start_date_dt - relativedelta(days=100)).strftime("%Y-%m-%d")
            get_insider_trades(ticker, self._end_date, start_date=insider_start, limit=1000)
            get_company_news(ticker, self._end_date, start_date=self._start_date, limit=1000)
        
        # Preload data for SPY for benchmark comparison
        get_prices("SPY", self._start_date, self._end_date)


    def run_backtest(self) -> PerformanceMetrics:
        self._prefetch_data()

        dates = pd.date_range(self._start_date, self._end_date, freq="B")
        if len(dates) > 0:
            self._portfolio_values = [
                {"Date": dates[0], "Portfolio Value": self._initial_capital}
            ]
        else:
            self._portfolio_values = []

        prev_trading_day: str | None = None
        for current_date in dates:
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 1) 取 T 日 K 線：開盤價用來成交、收盤價只用來估值。T 日休市則跳過。
            open_prices: Dict[str, float] = {}
            current_prices: Dict[str, float] = {}
            missing_data = False
            for ticker in self._tickers:
                try:
                    bar = get_price_data(ticker, current_date_str, current_date_str)
                    if bar.empty:
                        missing_data = True
                        break
                    last = bar.iloc[-1]
                    open_prices[ticker] = float(last["open"])
                    current_prices[ticker] = float(last["close"])
                except Exception:
                    missing_data = True
                    break
            if missing_data:
                continue

            # 2) 第一個交易日沒有「前一個交易日」可用 → 不交易，只估值。
            signal_date = prev_trading_day
            prev_trading_day = current_date_str
            if signal_date is None:
                continue

            # 3) 決策只看 T-1（含）以前的資料，往回 LOOKBACK_MONTHS 個月。
            lookback_start = (datetime.strptime(signal_date, "%Y-%m-%d") - relativedelta(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")
            agent_output = self._agent_controller.run_agent(
                self._agent,
                tickers=self._tickers,
                start_date=lookback_start,
                end_date=signal_date,
                portfolio=self._portfolio,
                model_name=self._model_name,
                model_provider=self._model_provider,
                selected_analysts=self._selected_analysts,
            )
            decisions = agent_output["decisions"]

            # 4) T 日開盤成交。
            executed_trades: Dict[str, int] = {}
            for ticker in self._tickers:
                d = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action = d.get("action", "hold")
                qty = d.get("quantity", 0)
                executed_qty = self._executor.execute_trade(ticker, action, qty, open_prices[ticker], self._portfolio)
                executed_trades[ticker] = executed_qty

            total_value = calculate_portfolio_value(self._portfolio, current_prices)
            exposures = compute_exposures(self._portfolio, current_prices)

            point: PortfolioValuePoint = {
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": exposures["Long Exposure"],
                "Short Exposure": exposures["Short Exposure"],
                "Gross Exposure": exposures["Gross Exposure"],
                "Net Exposure": exposures["Net Exposure"],
                "Long/Short Ratio": exposures["Long/Short Ratio"],
            }
            self._portfolio_values.append(point)
            
            # Build daily rows (stateless usage)
            rows = self._results.build_day_rows(
                date_str=current_date_str,
                tickers=self._tickers,
                agent_output=agent_output,
                executed_trades=executed_trades,
                current_prices=current_prices,
                portfolio=self._portfolio,
                performance_metrics=self._performance_metrics,
                total_value=total_value,
                benchmark_return_pct=self._benchmark.get_return_pct("SPY", self._start_date, current_date_str),
            )
            # Prepend today's rows to historical rows so latest day is on top
            self._table_rows = rows + self._table_rows
            # Print full history with latest day first (matches backtester.py behavior)
            self._results.print_rows(self._table_rows)

            # Update performance metrics after printing (match original timing)
            if len(self._portfolio_values) > 3:
                computed = self._perf.compute_metrics(self._portfolio_values)
                if computed:
                    self._performance_metrics.update(computed)

        self._performance_metrics["execution_timing"] = EXECUTION_TIMING
        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)


