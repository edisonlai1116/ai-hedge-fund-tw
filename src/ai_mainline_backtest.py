from __future__ import annotations

"""AI 主線投組長線回測 (AI Mainline Portfolio Long-term Backtest).

以 AI 產業鏈宇宙選股，採 6-18 個月波段（中長線）策略進行逐日投資組合模擬，
輸出以投報率(ROI)為核心的績效指標：累積報酬、年化報酬(CAGR)、最大回撤、
Sharpe、勝率、對標指數超額報酬，以及 AI 產業鏈分層貢獻與權益曲線。

刻意維持與 simple_signal / sp500_daily 一致的資料來源(yfinance)與技術指標，
不調整既有大架構，以新模組形式提供，方便 API 與前端取用。
"""

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# 回看期間排序（用於雲端夾上限，避免免費機跑太重逾時）。
_PERIOD_YEARS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "10y": 10, "max": 99}


def _clamp_period(period: str, max_period: str | None) -> str:
    """把回看期間夾到 max_period（雲端免費機用，由環境變數 AI_BACKTEST_MAX_PERIOD 設定）。"""
    if not max_period:
        return period
    want = _PERIOD_YEARS.get(str(period).lower(), 5)
    cap = _PERIOD_YEARS.get(str(max_period).lower(), 5)
    return max_period if want > cap else period

from src.simple_signal import (
    _DOWNLOAD_LOCK,
    compute_atr,
    compute_macd,
    compute_rsi,
    map_ai_chain_and_bottleneck,
    normalize_ticker,
)


# 預設 AI 產業鏈主線宇宙（皆可對應到 8 層 AI 產業鏈定位）。
AI_MAINLINE_UNIVERSE: dict[str, list[str]] = {
    "us": [
        "NVDA", "AVGO", "AMD", "TSM", "ASML", "AMAT", "LRCX", "KLAC",
        "MU", "ARM", "SNPS", "CDNS", "PLTR", "MSFT", "ANET", "VRT",
        "DELL", "CEG", "VST", "COHR",
    ],
    "tw": [
        "2330.TW", "2454.TW", "2317.TW", "2382.TW", "3017.TW", "3661.TW",
        "3443.TW", "2308.TW", "2345.TW", "3231.TW",
    ],
}

BENCHMARK_INDEX: dict[str, str] = {"us": "SPY", "tw": "^TWII"}

TRADING_DAYS_PER_YEAR = 252


@dataclass
class AiMainlineTrade:
    symbol: str
    layer: str | None
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    days_held: int
    outcome: str
    pnl: float


@dataclass
class LayerContribution:
    layer: str
    trades: int
    win_rate: float
    avg_return_pct: float
    net_pnl: float
    contribution_pct: float


@dataclass
class AiMainlineBacktestResult:
    market: str
    start_date: str
    end_date: str
    years: float
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    avg_trade_return_pct: float
    benchmark_symbol: str
    benchmark_return_pct: float
    excess_return_pct: float
    universe: list[str] = field(default_factory=list)
    layer_breakdown: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    trades_log: list[dict] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# 資料下載與技術指標
# ---------------------------------------------------------------------------
def _download_price_map(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    with _DOWNLOAD_LOCK:
        data = yf.download(
            symbols,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
    if data is None or data.empty:
        raise ValueError("無法下載 AI 主線回測所需的股價資料，請確認網路連線。")

    result: dict[str, pd.DataFrame] = {}
    required = {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(data.columns, pd.MultiIndex):
        level1 = set(data.columns.get_level_values(1))
        level0 = set(data.columns.get_level_values(0))
        for symbol in symbols:
            try:
                if symbol in level1:
                    frame = data.xs(symbol, axis=1, level=1).copy()
                elif symbol in level0:
                    frame = data[symbol].copy()
                else:
                    continue
                if required <= set(frame.columns):
                    result[symbol] = frame.dropna(subset=["High", "Low", "Close"]).copy()
            except Exception:
                continue
    else:
        frame = data.copy()
        if required <= set(frame.columns):
            result[symbols[0]] = frame.dropna(subset=["High", "Low", "Close"]).copy()
    return result


def _enrich(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["MA20"] = f["Close"].rolling(20).mean()
    f["MA50"] = f["Close"].rolling(50).mean()
    f["MA120"] = f["Close"].rolling(120).mean()
    f["RSI14"] = compute_rsi(f["Close"], 14)
    macd_line, signal_line, _ = compute_macd(f["Close"])
    f["MACD"] = macd_line
    f["MACD_Signal"] = signal_line
    f["ATR14"] = compute_atr(f, 14)
    return f


def _benchmark_return_pct(market: str, index: pd.DatetimeIndex) -> tuple[str, float]:
    benchmark_symbol = BENCHMARK_INDEX.get(market, "SPY")
    try:
        with _DOWNLOAD_LOCK:
            data = yf.download(
                benchmark_symbol,
                start=index[0].strftime("%Y-%m-%d"),
                end=index[-1].strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        if data is None or data.empty:
            return benchmark_symbol, 0.0
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 2:
            return benchmark_symbol, 0.0
        return benchmark_symbol, float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return benchmark_symbol, 0.0


# ---------------------------------------------------------------------------
# 投資組合模擬
# ---------------------------------------------------------------------------
def run_ai_mainline_backtest(
    *,
    symbols: list[str] | None = None,
    market: str = "us",
    period: str = "3y",
    initial_capital: float = 100000.0,
    max_positions: int = 8,
    take_profit_pct: float = 35.0,
    trailing_stop_pct: float = 18.0,
    max_holding_days: int = 126,
) -> dict[str, Any]:
    """執行 AI 主線投組波段回測，回傳可序列化的結果字典。

    策略（讓獲利奔跑、持有上限 6 個月；回測實測平均持有約 47 個交易日）：
    - 進場：MACD 黃金交叉，或多頭回檔（站上 MA20/MA50、RSI 40-62、MACD 多頭）。
    - 出場：達 +35% 目標報酬、自高點回落 18% 移動停利、跌破長線 MA120（持有>30日），
      或達 6 個月(126 交易日)最長持有期。
    - 資金：等權配置，最多同時持有 max_positions 檔，採市值動態切分。

    說明：經 5 年/10 年回測，「寬鬆停利(35%/18%)讓獲利奔跑」勝過「3–6 月緊縮(20%/10%)」；
    且持有上限 126 天(6 月)優於 378 天(18 月)——多數部位在達標/停損前 ~47 日就出場，
    縮短上限只清掉極少數卡住的爛單，年化更高、回撤略降。
    """
    market = "tw" if str(market).lower() == "tw" else "us"
    # 雲端免費機：把回看期間夾到上限（AI_BACKTEST_MAX_PERIOD，例如 5y），避免下載過久逾時回 HTML。
    period = _clamp_period(period, os.environ.get("AI_BACKTEST_MAX_PERIOD"))

    if symbols:
        universe = [normalize_ticker(s, market) for s in symbols if s and s.strip()]
    else:
        universe = list(AI_MAINLINE_UNIVERSE[market])
    # 去重並保序
    seen: set[str] = set()
    universe = [s for s in universe if not (s in seen or seen.add(s))]
    if not universe:
        raise ValueError("AI 主線回測需要至少一檔股票。")

    max_positions = max(1, min(int(max_positions), len(universe)))

    price_map = _download_price_map(universe, period)
    enriched: dict[str, pd.DataFrame] = {}
    layers: dict[str, str | None] = {}
    for symbol in universe:
        frame = price_map.get(symbol)
        if frame is None or len(frame) < 150:
            continue
        enriched[symbol] = _enrich(frame)
        layers[symbol], _ = map_ai_chain_and_bottleneck(symbol, "")

    if not enriched:
        raise ValueError("AI 主線回測的有效個股資料不足（需至少 150 個交易日）。")

    active_universe = list(enriched.keys())

    # 建立統一日期索引（所有個股交易日的聯集）。
    unified = sorted(set().union(*[set(f.index) for f in enriched.values()]))
    unified_index = pd.DatetimeIndex(unified)
    n = len(unified_index)

    # 各欄位對齊到統一索引的 numpy 陣列，缺值以 NaN 表示。
    cols = ["Close", "MA20", "MA50", "MA120", "RSI14", "MACD", "MACD_Signal"]
    aligned: dict[str, dict[str, np.ndarray]] = {}
    close_ff: dict[str, np.ndarray] = {}
    for symbol, f in enriched.items():
        r = f.reindex(unified_index)
        aligned[symbol] = {c: r[c].to_numpy(dtype=float) for c in cols}
        close_ff[symbol] = r["Close"].ffill().to_numpy(dtype=float)

    cash = float(initial_capital)
    positions: dict[str, dict[str, Any]] = {}
    trades: list[AiMainlineTrade] = []
    equity_dates: list[pd.Timestamp] = []
    equity_values: list[float] = []

    def holdings_value(i: int) -> float:
        total = 0.0
        for sym, pos in positions.items():
            px = close_ff[sym][i]
            if not np.isnan(px):
                total += pos["shares"] * px
        return total

    def _is_entry(sym: str, i: int) -> bool:
        a = aligned[sym]
        close, ma20, ma50 = a["Close"][i], a["MA20"][i], a["MA50"][i]
        rsi, macd, macd_sig = a["RSI14"][i], a["MACD"][i], a["MACD_Signal"][i]
        if any(np.isnan(v) for v in (close, ma20, ma50, rsi, macd, macd_sig)):
            return False
        prev_macd, prev_sig = a["MACD"][i - 1], a["MACD_Signal"][i - 1]
        macd_gold_cross = (
            not np.isnan(prev_macd)
            and not np.isnan(prev_sig)
            and macd > macd_sig
            and prev_macd <= prev_sig
        )
        bullish_pullback = (
            close > ma50 and close > ma20 and 40.0 <= rsi <= 62.0 and macd > macd_sig
        )
        return bool(macd_gold_cross or bullish_pullback)

    def _trend_strength(sym: str, i: int) -> float:
        a = aligned[sym]
        close, ma50 = a["Close"][i], a["MA50"][i]
        if np.isnan(close) or np.isnan(ma50) or ma50 <= 0:
            return -1e9
        return close / ma50 - 1.0

    tp_mult = 1.0 + take_profit_pct / 100.0
    trail_mult = 1.0 - trailing_stop_pct / 100.0

    for i in range(n):
        # 1) 出場檢查
        for sym in list(positions.keys()):
            px = close_ff[sym][i]
            if np.isnan(px):
                continue
            pos = positions[sym]
            pos["days_held"] += 1
            pos["peak"] = max(pos["peak"], px)
            trailing_stop = max(pos["stop_loss"], pos["peak"] * trail_mult)
            ma120 = aligned[sym]["MA120"][i]

            exit_reason: str | None = None
            if px >= pos["take_profit"]:
                exit_reason = "獲利了結 (+35%)"
            elif px <= trailing_stop:
                exit_reason = "移動停利/停損"
            elif (not np.isnan(ma120)) and px < ma120 and pos["days_held"] > 30:
                exit_reason = "跌破長線 MA120"
            elif pos["days_held"] >= max_holding_days:
                exit_reason = "達最長持有期"

            if exit_reason:
                proceeds = pos["shares"] * px
                cash += proceeds
                ret_pct = (px / pos["entry_price"] - 1.0) * 100.0
                trades.append(
                    AiMainlineTrade(
                        symbol=sym,
                        layer=layers.get(sym),
                        entry_date=pos["entry_date"],
                        exit_date=unified_index[i].strftime("%Y-%m-%d"),
                        entry_price=round(pos["entry_price"], 2),
                        exit_price=round(px, 2),
                        return_pct=round(ret_pct, 2),
                        days_held=pos["days_held"],
                        outcome=exit_reason,
                        pnl=round(proceeds - pos["cost"], 2),
                    )
                )
                del positions[sym]

        # 2) 進場檢查（填補空位，依趨勢強度排序挑選主線最強者）
        free_slots = max_positions - len(positions)
        if free_slots > 0:
            candidates = [
                sym
                for sym in active_universe
                if sym not in positions and _is_entry(sym, i)
            ]
            candidates.sort(key=lambda s: _trend_strength(s, i), reverse=True)
            equity_now = cash + holdings_value(i)
            slot_value = equity_now / max_positions
            for sym in candidates[:free_slots]:
                px = aligned[sym]["Close"][i]
                if np.isnan(px) or px <= 0:
                    continue
                spend = min(slot_value, cash)
                if spend < px:  # 連 1 股都買不起
                    continue
                shares = spend / px
                cost = shares * px
                cash -= cost
                positions[sym] = {
                    "shares": shares,
                    "entry_price": px,
                    "entry_date": unified_index[i].strftime("%Y-%m-%d"),
                    "cost": cost,
                    "peak": px,
                    "take_profit": px * tp_mult,
                    "stop_loss": px * trail_mult,
                    "days_held": 0,
                }

        equity_dates.append(unified_index[i])
        equity_values.append(cash + holdings_value(i))

    # 期末平倉，使交易統計完整（權益曲線已含未實現損益，不受影響）。
    last_i = n - 1
    for sym in list(positions.keys()):
        px = close_ff[sym][last_i]
        if np.isnan(px):
            continue
        pos = positions[sym]
        ret_pct = (px / pos["entry_price"] - 1.0) * 100.0
        trades.append(
            AiMainlineTrade(
                symbol=sym,
                layer=layers.get(sym),
                entry_date=pos["entry_date"],
                exit_date=unified_index[last_i].strftime("%Y-%m-%d"),
                entry_price=round(pos["entry_price"], 2),
                exit_price=round(px, 2),
                return_pct=round(ret_pct, 2),
                days_held=pos["days_held"],
                outcome="期末平倉結算",
                pnl=round(pos["shares"] * px - pos["cost"], 2),
            )
        )

    return _build_result(
        market=market,
        universe=active_universe,
        initial_capital=float(initial_capital),
        equity_dates=equity_dates,
        equity_values=equity_values,
        trades=trades,
        layers=layers,
    )


# ---------------------------------------------------------------------------
# 績效指標彙整
# ---------------------------------------------------------------------------
def _build_result(
    *,
    market: str,
    universe: list[str],
    initial_capital: float,
    equity_dates: list[pd.Timestamp],
    equity_values: list[float],
    trades: list[AiMainlineTrade],
    layers: dict[str, str | None],
) -> dict[str, Any]:
    equity = np.array(equity_values, dtype=float)
    start_dt, end_dt = equity_dates[0], equity_dates[-1]
    days = max((end_dt - start_dt).days, 1)
    years = days / 365.25

    final_equity = float(equity[-1])
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0
    cagr_pct = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_drawdown_pct = float(drawdowns.min() * 100.0) if len(drawdowns) else 0.0

    daily_ret = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    if daily_ret.size > 1 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    total_trades = len(trades)
    wins = [t for t in trades if t.return_pct >= 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades else 0.0
    avg_holding_days = (sum(t.days_held for t in trades) / total_trades) if total_trades else 0.0
    avg_trade_return_pct = (sum(t.return_pct for t in trades) / total_trades) if total_trades else 0.0

    benchmark_symbol, benchmark_return_pct = _benchmark_return_pct(
        market, pd.DatetimeIndex(equity_dates)
    )
    excess_return_pct = total_return_pct - benchmark_return_pct

    # 產業鏈分層貢獻
    total_net_pnl = sum(t.pnl for t in trades)
    layer_groups: dict[str, list[AiMainlineTrade]] = {}
    for t in trades:
        key = t.layer or "其他 / 未分類"
        layer_groups.setdefault(key, []).append(t)
    layer_breakdown: list[LayerContribution] = []
    for layer_name, group in layer_groups.items():
        net = sum(t.pnl for t in group)
        wr = sum(1 for t in group if t.return_pct >= 0) / len(group) * 100.0
        avg = sum(t.return_pct for t in group) / len(group)
        contribution = (net / total_net_pnl * 100.0) if total_net_pnl else 0.0
        layer_breakdown.append(
            LayerContribution(
                layer=layer_name,
                trades=len(group),
                win_rate=round(wr, 1),
                avg_return_pct=round(avg, 2),
                net_pnl=round(net, 2),
                contribution_pct=round(contribution, 1),
            )
        )
    layer_breakdown.sort(key=lambda x: x.net_pnl, reverse=True)

    # 權益曲線降採樣（最多 ~150 點，方便前端繪圖）
    step = max(1, len(equity_dates) // 150)
    curve = [
        {
            "date": equity_dates[idx].strftime("%Y-%m-%d"),
            "equity": round(float(equity[idx]), 2),
            "return_pct": round((float(equity[idx]) / initial_capital - 1.0) * 100.0, 2),
        }
        for idx in range(0, len(equity_dates), step)
    ]
    if curve and curve[-1]["date"] != end_dt.strftime("%Y-%m-%d"):
        curve.append(
            {
                "date": end_dt.strftime("%Y-%m-%d"),
                "equity": round(final_equity, 2),
                "return_pct": round(total_return_pct, 2),
            }
        )

    trades_sorted = sorted(trades, key=lambda t: t.exit_date)
    note = (
        f"以 {len(universe)} 檔 AI 產業鏈主線股、6-18 個月波段策略回測 {years:.1f} 年；"
        f"累積報酬 {total_return_pct:.1f}%、年化 {cagr_pct:.1f}%、"
        f"對標{benchmark_symbol} {'超額' if excess_return_pct >= 0 else '落後'} {abs(excess_return_pct):.1f}%。"
    )

    result = AiMainlineBacktestResult(
        market=market,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        years=round(years, 2),
        initial_capital=round(initial_capital, 2),
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        cagr_pct=round(cagr_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        sharpe_ratio=round(sharpe, 2),
        win_rate=round(win_rate, 1),
        total_trades=total_trades,
        avg_holding_days=round(avg_holding_days, 1),
        avg_trade_return_pct=round(avg_trade_return_pct, 2),
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=round(benchmark_return_pct, 2),
        excess_return_pct=round(excess_return_pct, 2),
        universe=universe,
        layer_breakdown=[asdict(x) for x in layer_breakdown],
        equity_curve=curve,
        trades_log=[asdict(t) for t in trades_sorted[-40:]],
        note=note,
    )
    return asdict(result)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AI 主線投組長線回測")
    parser.add_argument("--market", default="us", help="市場：us 或 tw")
    parser.add_argument("--period", default="3y", help="Yahoo Finance 回看期間，例如 3y / 5y")
    parser.add_argument("--tickers", default=None, help="自訂股票（逗號分隔）；省略則用預設 AI 主線宇宙")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--max-positions", type=int, default=8)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.tickers.split(",")] if args.tickers else None
    result = run_ai_mainline_backtest(
        symbols=symbols,
        market=args.market,
        period=args.period,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
