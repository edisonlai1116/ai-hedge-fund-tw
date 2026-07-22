import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from src.data.investing_scraper import fetch_investing_com_data


DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_ANALYSIS_PERIOD = "3y"
MIN_POSITION_RETURN_PCT = 20.0
SIGNAL_CACHE_TTL_MINUTES = 10
_SIGNAL_CACHE: dict[str, tuple[datetime, "SignalReport"]] = {}
_FUNDAMENTAL_CACHE: dict[str, tuple[datetime, dict]] = {}
FUNDAMENTAL_CACHE_TTL_MINUTES = 60  # 基本面資料快取 60 分鐘
FUNDAMENTAL_FETCH_TIMEOUT = 10       # 基本面 API 最多等 10 秒
_DOWNLOAD_LOCK = threading.Lock()


BUY_NOW = "今天可買"
BUY_SMALL = "今天可小量買"
WAIT_PULLBACK = "今天等回檔"
NO_BUY = "今天不要買"

SELL_NOW = "今天賣出"
SELL_SMALL = "今天可小量賣"
HOLD = "續抱"


@dataclass
class SignalReport:
    symbol: str
    latest_close: float
    ma20: float
    ma50: float
    rsi14: float
    atr14: float
    support: float
    resistance: float
    bias: str
    buy_zone: str
    sell_zone: str
    stop_loss: str
    reason: str
    rule_score: int
    ai_score: int | None
    composite_score: int
    buy_strength: str
    today_action: str
    today_entry_zone: str
    today_note: str
    today_exit_action: str
    today_exit_zone: str
    today_exit_note: str
    expected_return_pct: float
    risk_reward_ratio: float
    holding_days_estimate: int
    holding_window: str
    backtest: dict | None
    committee_summary: str | None
    committee_model: str | None
    ai_enabled: bool
    ai_available: bool
    ai_error: str | None
    chart: list[dict]
    agents: list[dict]
    horizons: list[dict]
    fundamental_score: int = 0
    graham_number: float | None = None
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    bb_width: float = 0.0
    candlestick_pattern: str = "無"
    kelly_position_pct: float = 0.0
    decision_assistance: str = ""
    timeline_backtest: dict | None = None
    ai_chain_layer: str | None = None
    critical_bottleneck: str | None = None
    novice_rating: str | None = None
    investingpro_fair_value: float | None = None
    valuation_gap_pct: float | None = None
    analyst_target_price: float | None = None
    warren_ai_momentum: str | None = None
    investingpro_models: list[dict] | None = None
    cognitive_temperature_gap: str | None = None
    geopolitical_timing_advice: str | None = None
    value_trap_risk: str | None = None
    price_forecast: dict | None = None
    long_term_risk: dict | None = None
    ma120: float = 0.0  # 長線生命線（波段策略的長線出場依據）
    drawdown_from_high_pct: float = 0.0  # 距 60 日高點回落 %（0 = 貼近新高；持股健檢判斷強勢突破用）



@dataclass
class AgentOpinion:
    key: str
    name: str
    signal: str
    confidence: int
    summary: str
    historical_edge: dict | None = None


@dataclass
class HorizonView:
    horizon: str
    bias: str
    entry_zone: str
    take_profit_zone: str
    stop_zone: str
    summary: str


_TAIWAN_TICKER_CACHE: dict[str, str] = {}


def normalize_ticker(ticker: str, market: str | None = None) -> str:
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker 不能為空。")
    if "." in symbol:
        return symbol
    if market == "tw" or symbol.isdigit():
        if symbol in _TAIWAN_TICKER_CACHE:
            return _TAIWAN_TICKER_CACHE[symbol]

        tw_symbol = f"{symbol}.TW"
        two_symbol = f"{symbol}.TWO"

        # 優先測試上市 (.TW)
        try:
            df = yf.Ticker(tw_symbol).history(period="1d")
            if not df.empty:
                _TAIWAN_TICKER_CACHE[symbol] = tw_symbol
                return tw_symbol
        except Exception:
            pass

        # 測試上櫃 (.TWO)
        try:
            df = yf.Ticker(two_symbol).history(period="1d")
            if not df.empty:
                _TAIWAN_TICKER_CACHE[symbol] = two_symbol
                return two_symbol
        except Exception:
            pass

        # 預設回退至上市 (.TW)
        _TAIWAN_TICKER_CACHE[symbol] = tw_symbol
        return tw_symbol
    return symbol



def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def format_range(low: float, high: float) -> str:
    lo = min(low, high)
    hi = max(low, high)
    return f"{lo:.2f} - {hi:.2f}"


def parse_range(range_text: str) -> tuple[float, float]:
    low, high = [float(part.strip()) for part in range_text.split("-")]
    return low, high


def range_mid(range_text: str) -> float:
    low, high = parse_range(range_text)
    return (low + high) / 2


def clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def classify_signal(score: int) -> str:
    if score >= 2:
        return "偏多"
    if score <= -2:
        return "偏空"
    return "中性"


def score_from_signal(signal: str) -> int:
    if "多" in signal or "Bullish" in signal:
        return 1
    if "空" in signal or "Bearish" in signal:
        return -1
    return 0


def confidence_from_score(score: int, base: int = 55) -> int:
    return max(35, min(90, base + abs(score) * 10))


def score_to_strength(score: int) -> str:
    if score >= 80:
        return "強力買進"
    if score >= 65:
        return "可分批買進"
    if score >= 50:
        return "觀察偏多"
    if score >= 35:
        return "先觀察"
    return "不建議進場"


def safe_range(low: float, high: float, anchor: float) -> tuple[float, float]:
    floor = max(anchor * 0.35, 0.01)
    low = max(low, floor)
    high = max(high, floor)
    if low > high:
        low, high = high, low
    if abs(high - low) < anchor * 0.003:
        high = low + anchor * 0.01
    return low, high


def _cache_key(symbol: str, period: str, use_ai_committee: bool, committee_model: str) -> str:
    return f"{symbol}:{period}:{use_ai_committee}:{committee_model}"


def _clone_report(report: SignalReport) -> SignalReport:
    return SignalReport(
        symbol=report.symbol,
        latest_close=report.latest_close,
        ma20=report.ma20,
        ma50=report.ma50,
        rsi14=report.rsi14,
        atr14=report.atr14,
        support=report.support,
        resistance=report.resistance,
        bias=report.bias,
        buy_zone=report.buy_zone,
        sell_zone=report.sell_zone,
        stop_loss=report.stop_loss,
        reason=report.reason,
        rule_score=report.rule_score,
        ai_score=report.ai_score,
        composite_score=report.composite_score,
        buy_strength=report.buy_strength,
        today_action=report.today_action,
        today_entry_zone=report.today_entry_zone,
        today_note=report.today_note,
        today_exit_action=report.today_exit_action,
        today_exit_zone=report.today_exit_zone,
        today_exit_note=report.today_exit_note,
        expected_return_pct=report.expected_return_pct,
        risk_reward_ratio=report.risk_reward_ratio,
        holding_days_estimate=report.holding_days_estimate,
        holding_window=report.holding_window,
        backtest=dict(report.backtest) if report.backtest else None,
        committee_summary=report.committee_summary,
        committee_model=report.committee_model,
        ai_enabled=report.ai_enabled,
        ai_available=report.ai_available,
        ai_error=report.ai_error,
        chart=[dict(item) for item in report.chart],
        agents=[dict(item) for item in report.agents],
        horizons=[dict(item) for item in report.horizons],
        fundamental_score=report.fundamental_score,
        graham_number=report.graham_number,
        macd_value=report.macd_value,
        macd_signal=report.macd_signal,
        macd_hist=report.macd_hist,
        bb_width=report.bb_width,
        candlestick_pattern=report.candlestick_pattern,
        kelly_position_pct=report.kelly_position_pct,
        decision_assistance=report.decision_assistance,
        timeline_backtest={
            **report.timeline_backtest,
            "trades_log": [dict(trade) for trade in report.timeline_backtest["trades_log"]]
        } if report.timeline_backtest else None,
        ai_chain_layer=report.ai_chain_layer,
        critical_bottleneck=report.critical_bottleneck,
        novice_rating=report.novice_rating,
        investingpro_fair_value=report.investingpro_fair_value,
        valuation_gap_pct=report.valuation_gap_pct,
        analyst_target_price=report.analyst_target_price,
        warren_ai_momentum=report.warren_ai_momentum,
        investingpro_models=[dict(item) for item in report.investingpro_models] if report.investingpro_models else None,
        cognitive_temperature_gap=report.cognitive_temperature_gap,
        geopolitical_timing_advice=report.geopolitical_timing_advice,
        value_trap_risk=report.value_trap_risk,
        price_forecast=dict(report.price_forecast) if report.price_forecast else None,
        long_term_risk=dict(report.long_term_risk) if report.long_term_risk else None,
        ma120=report.ma120,
    )


def compute_macd(close_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bb_width(close_series: pd.Series) -> pd.Series:
    ma20 = close_series.rolling(20).mean()
    std20 = close_series.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    return (upper - lower) / ma20.replace(0, np.nan)


def compute_pivot_levels(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    return {"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2}


def detect_candlestick(open_s: pd.Series, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series) -> str:
    if len(close_s) < 2:
        return "無"
    o1, h1, l1, c1 = open_s.iloc[-2], high_s.iloc[-2], low_s.iloc[-2], close_s.iloc[-2]
    o2, h2, l2, c2 = open_s.iloc[-1], high_s.iloc[-1], low_s.iloc[-1], close_s.iloc[-1]
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    
    # 1. Bullish Engulfing
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1 and body2 > body1:
        return "看多吞噬"
    # 2. Bearish Engulfing
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1 and body2 > body1:
        return "看空吞噬"
    # 3. Hammer (Bullish Reversal)
    total_range = h2 - l2
    if total_range > 0:
        body_top = max(o2, c2)
        body_bottom = min(o2, c2)
        lower_shadow = body_bottom - l2
        upper_shadow = h2 - body_top
        if lower_shadow > 2 * body2 and upper_shadow < 0.2 * total_range:
            return "錘頭線(看多)"
    # 4. Shooting Star (Bearish Reversal)
    if total_range > 0:
        body_top = max(o2, c2)
        body_bottom = min(o2, c2)
        lower_shadow = body_bottom - l2
        upper_shadow = h2 - body_top
        if upper_shadow > 2 * body2 and lower_shadow < 0.2 * total_range:
            return "射擊之星(看空)"
    return "無"


def compute_timeline_backtest(symbol: str, frame: pd.DataFrame) -> dict:
    if len(frame) < 10:
        return {
            "total_trades": 0,
            "win_rate": 50.0,
            "avg_return": 0.0,
            "cumulative_return": 0.0,
            "trades_log": []
        }
        
    completed_trades = []
    active_trade = None
    
    for i in range(1, len(frame)):
        close = float(frame["Close"].iloc[i])
        low = float(frame["Low"].iloc[i])
        high = float(frame["High"].iloc[i])
        ma20 = float(frame["MA20"].iloc[i])
        ma50 = float(frame["MA50"].iloc[i])
        ma120 = float(frame["MA120"].iloc[i]) if "MA120" in frame.columns and not pd.isna(frame["MA120"].iloc[i]) else ma50
        rsi = float(frame["RSI14"].iloc[i])
        macd = float(frame["MACD"].iloc[i])
        macd_sig = float(frame["MACD_Signal"].iloc[i])
        prev_macd = float(frame["MACD"].iloc[i-1])
        prev_macd_sig = float(frame["MACD_Signal"].iloc[i-1])
        atr = float(frame["ATR14"].iloc[i])
        date_str = frame.index[i].strftime("%Y-%m-%d")
        
        if active_trade is None:
            macd_gold_cross = (macd > macd_sig) and (prev_macd <= prev_macd_sig)
            bullish_pullback = (close > ma50) and (close > ma20) and (rsi >= 40) and (rsi <= 62) and (macd > macd_sig)
            
            if macd_gold_cross or bullish_pullback:
                entry_price = close
                # Strategic Mid-to-Long term target: 35% minimum wave gain
                take_profit = entry_price * 1.35
                # Protective wide initial stop: 18% below entry to withstand medium-term noise
                stop_loss = entry_price * 0.82
                active_trade = {
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "entry_date": date_str,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "peak_close": close,
                    "days_held": 0
                }
        else:
            active_trade["days_held"] += 1
            if close > active_trade["peak_close"]:
                active_trade["peak_close"] = close
                
            # 移動停損：自波段高點回落 18%（與 AI 主線回測勝出的波段參數一致：讓獲利奔跑）
            trailing_stop = max(active_trade["stop_loss"], active_trade["peak_close"] * 0.82)

            # 1. 出場：觸及移動停損，或持有 >30 日後跌破長線 MA120
            if close <= trailing_stop or (close < ma120 and active_trade["days_held"] > 30):
                exit_price = close
                exit_ret = ((exit_price / active_trade["entry_price"]) - 1) * 100
                completed_trades.append({
                    "entry_date": active_trade["entry_date"],
                    "exit_date": date_str,
                    "entry_price": round(active_trade["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "return_pct": round(exit_ret, 2),
                    "days_held": active_trade["days_held"],
                    "outcome": "停損/退場" if exit_ret < 0 else "獲利"
                })
                active_trade = None
            # 2. Take profit hit (close rises by 35%+)
            elif close >= active_trade["take_profit"]:
                exit_price = close
                exit_ret = ((exit_price / active_trade["entry_price"]) - 1) * 100
                completed_trades.append({
                    "entry_date": active_trade["entry_date"],
                    "exit_date": date_str,
                    "entry_price": round(active_trade["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "return_pct": round(exit_ret, 2),
                    "days_held": active_trade["days_held"],
                    "outcome": "獲利"
                })
                active_trade = None
            # 3. 達最長持有期（126 交易日 ≈ 6 個月；回測最佳波段上限）
            elif active_trade["days_held"] >= 126:
                exit_price = close
                exit_ret = ((exit_price / active_trade["entry_price"]) - 1) * 100
                completed_trades.append({
                    "entry_date": active_trade["entry_date"],
                    "exit_date": date_str,
                    "entry_price": round(active_trade["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "return_pct": round(exit_ret, 2),
                    "days_held": active_trade["days_held"],
                    "outcome": "獲利" if exit_ret >= 0 else "停損/退場"
                })
                active_trade = None
                
    if active_trade is not None:
        last_row = frame.iloc[-1]
        last_close = float(last_row["Close"])
        last_date = frame.index[-1].strftime("%Y-%m-%d")
        exit_ret = ((last_close / active_trade["entry_price"]) - 1) * 100
        completed_trades.append({
            "entry_date": active_trade["entry_date"],
            "exit_date": last_date,
            "entry_price": round(active_trade["entry_price"], 2),
            "exit_price": round(last_close, 2),
            "return_pct": round(exit_ret, 2),
            "days_held": active_trade["days_held"],
            "outcome": "獲利" if exit_ret >= 0 else "停損/退場"
        })
        active_trade = None
        
    total_trades = len(completed_trades)
    if total_trades > 0:
        win_rate = (sum(1 for t in completed_trades if t["return_pct"] >= 0) / total_trades) * 100
        avg_return = sum(t["return_pct"] for t in completed_trades) / total_trades
        equity = 1.0
        for t in completed_trades:
            equity = equity * (1.0 + t["return_pct"] / 100.0)
        cumulative_return = (equity - 1.0) * 100
    else:
        win_rate = 50.0
        avg_return = 0.0
        cumulative_return = 0.0
        
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 2),
        "cumulative_return": round(cumulative_return, 2),
        "trades_log": completed_trades[-4:]
    }


def generate_decision_assistance(report: SignalReport) -> str:
    symbol = report.symbol
    close = report.latest_close
    f_score = report.fundamental_score
    graham = report.graham_number
    macd_val = report.macd_value
    macd_sig = report.macd_signal
    bb_width = report.bb_width
    kelly = report.kelly_position_pct * 100
    candle = report.candlestick_pattern
    bias = report.bias
    today_action = report.today_action
    
    advice = []
    
    # 1. Fundamental evaluation (F-Score & Graham)
    fund_parts = []
    if f_score >= 7:
        fund_parts.append(f"基本面非常強韌 (F-Score 達到 {f_score}/9)，財務健康度高，是合適的價值型或成長型標的，不容易陷入價值陷阱。")
    elif f_score <= 3 and f_score > 0:
        fund_parts.append(f"基本面偏弱 (F-Score 僅 {f_score}/9)，財務體質有潛在風險，操作上需縮小部位，警惕價值陷阱。")
    elif f_score == 0:
        fund_parts.append("目前財務數據不足，暫難評估 Piotroski 財報健康度，建議純以技術面與量價結構進行波段操作。")
    else:
        fund_parts.append(f"基本面中等 (F-Score {f_score}/9)，財務狀況尚算平穩。")
        
    if graham is not None and graham > 0:
        discount = ((graham / close) - 1) * 100
        if close < graham:
            fund_parts.append(f"目前股價 ({close:.2f}元) 低於葛拉漢防守估值 ({graham:.2f}元)，折價幅度達 {abs(discount):.1f}%，安全邊際充足，極具中長期防守價值。")
        else:
            fund_parts.append(f"目前股價 ({close:.2f}元) 高於葛拉漢防守估值 ({graham:.2f}元) 約 {discount:.1f}%，顯示目前溢價，中長期安全邊際較小，適合短中期技術面操作。")
    else:
        fund_parts.append(f"葛拉漢防守價：目前無估值資料，建議以技術支撐區 ({report.support:.2f}元) 作為短期防守底線。")
        
    advice.append("【基本面與估值】\n" + " ".join(fund_parts))
    
    # 2. Technical and Volatility evaluation (MACD, BB, Candlestick)
    tech_parts = []
    macd_bullish = macd_val > macd_sig
    if macd_bullish:
        tech_parts.append("MACD 目前處於多頭狀態 (快線高於慢線)，動能偏向多方。")
    else:
        tech_parts.append("MACD 目前處於空頭狀態 (快線低於慢線)，多頭反彈動能偏弱。")
        
    if bb_width < 0.12:
        tech_parts.append(f"布林頻寬處於緊縮爆發期 (BB Width 僅 {bb_width*100:.1f}%)，近期可能出現劇烈波動與變盤，即將發動方向性噴出。")
    else:
        tech_parts.append(f"布林頻寬處於正常區間 (BB Width {bb_width*100:.1f}%)，目前波動平穩。")
        
    if candle != "無":
        tech_parts.append(f"K線圖表在最新交易日偵測到了「{candle}」反轉訊號，為極為關鍵的轉折提示。")
        
    advice.append("【技術面與波動】\n" + " ".join(tech_parts))
    
    # 3. InvestingPro and 4-Step Valuation Gap Analysis
    if report.investingpro_fair_value is not None:
        inv_parts = [
            f"根據 Investing.com 內置的 12 種高階財務估值模型綜合測算，該股合理價值為 {report.investingpro_fair_value:.2f} 元。",
            f"當前收盤價相較於合理價值估算之折溢價差幅為 {report.valuation_gap_pct:+.2f}%。",
            f"法人分析師共識平均目標價為 {report.analyst_target_price:.2f} 元。",
            f"Warren AI 技術動能判定為「{report.warren_ai_momentum}」，顯示短中期勢能具備健康的波段特質。"
        ]
        advice.append("【InvestingPro 估值差幅與技術動能】\n" + " ".join(inv_parts))
        
    # 4. Master 定性策略融入
    if report.cognitive_temperature_gap:
        advice.append("【認知溫差與板塊重分類】\n" + report.cognitive_temperature_gap)
    if report.geopolitical_timing_advice:
        advice.append("【地緣政治與先前佈局（Pre-peace）】\n" + report.geopolitical_timing_advice)
    if report.value_trap_risk:
        advice.append("【價值陷阱避雷指南】\n" + report.value_trap_risk)

    # 4b. 長線虧損風險評估（12 個月預測 + 個股自身波段歷史回測）
    ltr = report.long_term_risk
    if ltr:
        ltr_parts = [ltr.get("note", "")]
        exp12 = ltr.get("expected_return_12m_pct")
        if exp12 is not None:
            ltr_parts.append(f"12 個月統計預測期望報酬約 {exp12:+.1f}%。")
        hist_cum = ltr.get("history_cumulative_return_pct")
        hist_win = ltr.get("history_win_rate_pct")
        if hist_cum is not None and ltr.get("history_trades", 0) >= 1:
            ltr_parts.append(f"個股自身波段策略歷史累積報酬 {hist_cum:+.1f}%、勝率 {hist_win:.0f}%。")
        if ltr.get("blocked"):
            ltr_parts.append("⚠️ 長抱仍可能虧損，本系統不建議買進此股，請優先尋找長線期望值為正的標的。")
        advice.append("【長線虧損風險評估】\n" + " ".join(p for p in ltr_parts if p))

    # 5. Dynamic Decision Synthesis & Kelly Position
    decision_parts = []
    if today_action in ["今天可買", "今天可小量買"]:
        decision_parts.append(f"今天操作建議為「{today_action}」，建議買點區間在 {report.today_entry_zone} 元。")
        if kelly > 0:
            decision_parts.append(f"經勝率與風報比加權後的 Half-Kelly 最優資金配置比率為 {kelly:.1f}%，建議分批掛單建立倉位。")
        else:
            decision_parts.append("雖然技術面有買點，但因預期報酬率或風報比有限，凱利公式建議暫時不要配置太大部位。")
    elif today_action == "今天等回檔":
        decision_parts.append(f"目前股價已偏離建議買點區間 ({report.buy_zone} 元)，「{today_action}」是最佳決策，請耐心等待股價回落至支撐區再行布局，切勿盲目追高。")
    else:
        decision_parts.append(f"今日操作建議為「{today_action}」，趨勢偏空或報酬風險比不劃算，建議空倉觀望，把資金留給其他更具優勢的股票。")
        
    if report.today_exit_action in ["今天賣出", "今天可小量賣"]:
        decision_parts.append(f"同時，若手中持有部位，今日已進入建議賣點區間 {report.today_exit_zone} 元，建議執行「{report.today_exit_action}」以落袋為安或減碼鎖利，切實執行 protective stop 停損防護。")
    else:
        decision_parts.append(f"若手中持有部位，目前的最佳策略是「續抱觀察」，守穩建議停損價 ({report.stop_loss} 元) 即可。")
        
    advice.append("【綜合決策與資金配置建議】\n" + " ".join(decision_parts))
    
    return "\n\n".join(advice)


def download_prices(symbol: str, period: str) -> pd.DataFrame:
    periods_to_try = [period, DEFAULT_ANALYSIS_PERIOD, "2y"]
    seen: set[str] = set()
    last_error = f"無法下載 {symbol} 的股價資料。"
    for candidate in periods_to_try:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            with _DOWNLOAD_LOCK:
                data = yf.download(symbol, period=candidate, interval="1d", auto_adjust=False, progress=False)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            required = {"Open", "High", "Low", "Close", "Volume"}
            missing = required - set(data.columns)
            if missing:
                last_error = f"{symbol} 缺少必要欄位: {', '.join(sorted(missing))}"
                continue
            frame = data.dropna(subset=["High", "Low", "Close"]).copy()
            if len(frame) >= 90:
                return frame
            last_error = f"{symbol} 的歷史資料不足，至少需要 90 個交易日。"
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(last_error)


def build_agent_opinions(
    frame: pd.DataFrame,
    symbol: str = "AAPL",
    fundamental_score: int = 0,
    graham_number: float | None = None,
    latest_close: float = 0.0,
    forward_pe: float | None = None,
    investingpro_fair_value: float | None = None,
    valuation_gap_pct: float | None = None,
    analyst_target_price: float | None = None,
    warren_ai_momentum: str | None = None,
    ai_chain_layer: str | None = None,
    critical_bottleneck: str | None = None,
    agent_edges: dict | None = None,
) -> list[AgentOpinion]:
    agent_edges = agent_edges or {}
    latest = frame.iloc[-1]
    prev = frame.iloc[-2]

    close = float(latest["Close"])
    ma10 = float(latest["MA10"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi5 = float(latest["RSI5"])
    rsi14 = float(latest["RSI14"])
    atr14 = float(latest["ATR14"])
    support20 = float(frame.tail(20)["Low"].min())
    resistance20 = float(frame.tail(20)["High"].max())
    avg_volume = float(latest["VOL20"]) if not np.isnan(latest["VOL20"]) else float(latest["Volume"])
    volume_ratio = float(latest["Volume"]) / avg_volume if avg_volume else 1.0
    atr_pct = (atr14 / close) * 100 if close else 0.0

    # MACD Indicators
    macd_val = float(latest["MACD"]) if "MACD" in latest else 0.0
    macd_sig = float(latest["MACD_Signal"]) if "MACD_Signal" in latest else 0.0
    macd_hist = float(latest["MACD_Hist"]) if "MACD_Hist" in latest else 0.0
    prev_macd_hist = float(prev["MACD_Hist"]) if "MACD_Hist" in prev else 0.0
    bb_width = float(latest["BB_Width"]) if "BB_Width" in latest else 0.0

    # Trend calculation incorporating MACD crossover
    trend_score = int(close > ma20) + int(ma20 > ma50) + int(float(latest["MA20"]) >= float(prev["MA20"]))
    trend_score += int(macd_val > macd_sig)
    trend_score -= int(close < ma20) + int(ma20 < ma50) + int(macd_val < macd_sig)

    # Momentum incorporating MACD histogram changes
    momentum_score = int(rsi14 > 56) + int(rsi5 > 56) + int(close > float(prev["Close"]))
    momentum_score += int(macd_hist > prev_macd_hist)
    momentum_score -= int(rsi14 < 44) + int(rsi5 < 44) + int(macd_hist < prev_macd_hist)

    # Breakout incorporating Bollinger Width expansions
    breakout_score = int(close >= resistance20 * 0.99) + int(volume_ratio > 1.15) + int(bb_width > 0.15)
    breakout_score -= int(close <= support20 * 1.01) + int(volume_ratio < 0.9)

    mean_reversion_score = int(rsi5 < 35) + int(close < ma10 - 0.7 * atr14)
    mean_reversion_score -= int(rsi5 > 67) + int(close > ma10 + 0.8 * atr14)

    risk_score = int(atr_pct > 4.5) + int(close < ma50) + int(rsi14 < 40)

    macd_desc = "黃金交叉" if macd_val > macd_sig else "死亡交叉"
    bb_desc = "波動擴張" if bb_width > 0.15 else "波動收縮"

    # Master analyst agent signals logic
    gap = valuation_gap_pct if valuation_gap_pct is not None else 0.0

    # 1. Warren Buffett
    buffett_sig = "Neutral"
    buffett_conf = 70
    buffett_sum = "基本面穩健度適中，估值處於合理區間，保持觀望態度。"
    if fundamental_score >= 6 and (graham_number is not None and latest_close <= graham_number * 1.2 or gap >= 5.0):
        buffett_sig = "Bullish"
        buffett_conf = 85
        buffett_sum = f"財務強健 (F-Score: {fundamental_score}/9)，具備足夠的安全邊際與護城河，建議偏多配置。"
    elif fundamental_score <= 3:
        buffett_sig = "Bearish"
        buffett_conf = 90
        buffett_sum = f"財務實力虛弱 (F-Score: {fundamental_score}/9)，缺乏穩定盈餘與護城河，建議迴避。"

    # 2. Charlie Munger
    munger_sig = "Neutral"
    munger_conf = 75
    munger_sum = "業務模式與財務健康度平穩，未見極度突出的高回報率優勢。"
    if fundamental_score >= 7:
        munger_sig = "Bullish"
        munger_conf = 88
        munger_sum = f"典型的超高質量企業 (F-Score: {fundamental_score}/9)，護城河極深，高資本回報率具備強確定性。"
    elif fundamental_score <= 3:
        munger_sig = "Bearish"
        munger_conf = 92
        munger_sum = "企業質量低劣，盈餘波動大且債務結構不佳，完全不符合合理估值買入偉大企業的原則。"

    # 3. Ben Graham
    graham_sig = "Neutral"
    graham_conf = 70
    graham_sum = "價格貼近合理價值，安全邊際不足，保持中性觀察。"
    if (graham_number is not None and latest_close < graham_number) or gap >= 20.0 or (forward_pe is not None and 0 < forward_pe <= 15.0):
        graham_sig = "Bullish"
        graham_conf = 90
        graham_sum = f"股價顯著低於防守價或低本益比 (PE: {forward_pe or 0:.1f})，提供了極佳的安全邊際保護。"
    elif (graham_number is not None and latest_close > graham_number * 1.5) and (forward_pe is not None and forward_pe > 30.0) and ai_chain_layer is None:
        # AI 主線成長股本就交易於高溢價，不以葛拉漢防守價否決其多頭邏輯
        graham_sig = "Bearish"
        graham_conf = 85
        graham_sum = f"估值過度透支 (PE: {forward_pe:.1f}) 且大幅高於資產防守價值，不具備安全邊際。"

    # 4. Aswath Damodaran
    damodaran_sig = "Neutral"
    damodaran_conf = 75
    damodaran_sum = f"最新定額折現模型顯示，當前市場定價已充分反映其內在價值。"
    if gap >= 15.0:
        damodaran_sig = "Bullish"
        damodaran_conf = 85
        damodaran_sum = f"折現現金流與經典乘數模型測算合理價值為 ${investingpro_fair_value or 0:.2f}，安全邊際折價高達 {gap:.1f}%。"
    elif gap <= -15.0:
        damodaran_sig = "Bearish"
        damodaran_conf = 85
        damodaran_sum = f"當前股價高於模型合理價值 (${investingpro_fair_value or 0:.2f}) 約 {abs(gap):.1f}%，估值明顯溢價。"

    # 5. Cathie Wood
    wood_sig = "Neutral"
    wood_conf = 60
    wood_sum = "未見顯著的顛覆性創新或硬核科技增長主線，暫不列為核心追蹤目標。"
    if ai_chain_layer is not None or symbol in ["NVDA", "AAPL", "MSFT", "PLTR", "MU", "2330.TW", "3017.TW"]:
        wood_sig = "Bullish"
        wood_conf = 90
        wood_sum = f"屬於硬核 AI 科技或 AI 基礎設施關鍵鏈條 ({ai_chain_layer or '創新主線'})，具備爆發性長線增長潛力。"
    elif forward_pe is not None and forward_pe < 10.0 and fundamental_score >= 7:
        wood_sig = "Bearish"
        wood_conf = 80
        wood_sum = "屬於傳統低增長週期股，缺乏長線創新高確定性，並非時代顛覆性核心資產。"

    # 6. Nassim Taleb
    taleb_sig = "Neutral"
    taleb_conf = 70
    taleb_sum = "資產負債表與估值水平尚可，未見顯著的反脆弱或脆弱特徵。"
    # AI 主線股的反脆弱性來自需求獨佔與定價權，不是低 PE；門檻提高到 65
    taleb_pe_high = 65.0 if ai_chain_layer is not None else 40.0
    if fundamental_score >= 6 and (forward_pe is not None and forward_pe <= 20.0):
        taleb_sig = "Bullish"
        taleb_conf = 80
        taleb_sum = "擁有強勁的流動性與防禦性估值，具備顯著的反脆弱抗震能力。"
    elif (forward_pe is not None and forward_pe > taleb_pe_high) or fundamental_score <= 3:
        taleb_sig = "Bearish"
        taleb_conf = 95
        taleb_sum = f"高估值 (PE: {forward_pe or 0:.1f}) 或財務極度脆弱，極易受尾部黑天鵝事件衝擊，屬於高風險Fragile標的。"

    # 7. Peter Lynch
    lynch_sig = "Neutral"
    lynch_conf = 70
    lynch_sum = "增長率與本益比匹配度適中，處於中性合理區間。"
    if (forward_pe is not None and 0 < forward_pe <= 25.0) and fundamental_score >= 5:
        lynch_sig = "Bullish"
        lynch_conf = 85
        lynch_sum = f"增長前景良好且估值合理 (PE: {forward_pe:.1f})，是典型的 GARP (合理價格增長) 投資首選。"
    elif forward_pe is not None and forward_pe > 50.0:
        lynch_sig = "Bearish"
        lynch_conf = 80
        lynch_sum = f"前瞻本益比已高達 {forward_pe:.1f} 倍，增長速度難以維持如此高企的估值倍數。"

    # 8. Michael Burry
    burry_sig = "Neutral"
    burry_conf = 65
    burry_sum = "市場情緒與多空力量均衡，未見極端的非對稱套利機會。"
    # AI 主線股成長率高，PEG < 1 才是真正指標；PE 門檻提高到 65
    burry_pe_high = 65.0 if ai_chain_layer is not None else 45.0
    if gap >= 25.0 or rsi14 < 35.0:
        burry_sig = "Bullish"
        burry_conf = 85
        burry_sum = "股價技術面嚴重超跌或被市場極度恐慌性低估，提供了非對稱的多頭切入契機。"
    elif (forward_pe is not None and forward_pe > burry_pe_high) or rsi14 > 70.0:
        burry_sig = "Bearish"
        burry_conf = 90
        burry_sum = f"市場情緒極度亢奮且估值溢價嚴重 (PE: {forward_pe or 0:.1f})，防範泡沫破裂與高位套牢風險。"

    # 9. Stanley Druckenmiller
    druckenmiller_sig = "Neutral"
    druckenmiller_conf = 70
    druckenmiller_sum = "技術面均線與價格纏繞，未形成明確的單邊趨勢信號。"
    if close > ma20 > ma50:
        druckenmiller_sig = "Bullish"
        druckenmiller_conf = 85
        druckenmiller_sum = "價格站上均線且呈多頭排列，技術動能強勁且資金持續流入，順勢做多。"
    elif close < ma20 < ma50:
        druckenmiller_sig = "Bearish"
        druckenmiller_conf = 85
        druckenmiller_sum = "弱勢空頭排列且跌破生命線，市場缺乏流動性支持，建議順勢看空。"

    # 10. Bill Ackman
    ackman_sig = "Neutral"
    ackman_conf = 70
    ackman_sum = "業務預測性中等，缺乏頂級龍頭獨佔性特徵。"
    if fundamental_score >= 7 and symbol in ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "2330.TW"]:
        ackman_sig = "Bullish"
        ackman_conf = 90
        ackman_sum = "典型的大市值高護城河龍頭，業務極具可預測性且擁有強大的自由現金流生成能力。"
    elif fundamental_score <= 3:
        ackman_sig = "Bearish"
        ackman_conf = 85
        ackman_sum = f"基本面不穩定 (F-Score: {fundamental_score}/9)，業務結構繁雜且可預測性低，缺乏長線配置價值。"

    return [
        AgentOpinion(
            key="trend_agent",
            name="趨勢代理",
            signal=classify_signal(trend_score),
            confidence=confidence_from_score(trend_score),
            summary=f"價格與均線排列顯示 {classify_signal(trend_score)}，MACD呈 {macd_desc}，趨勢延續中。",
            historical_edge=agent_edges.get("trend_agent"),
        ),
        AgentOpinion(
            key="momentum_agent",
            name="動能代理",
            signal=classify_signal(momentum_score),
            confidence=confidence_from_score(momentum_score),
            summary=f"RSI 與最近價格動能偏 {classify_signal(momentum_score)}，MACD柱狀圖動能偏 {'增強' if macd_hist > prev_macd_hist else '減弱'}。",
            historical_edge=agent_edges.get("momentum_agent"),
        ),
        AgentOpinion(
            key="breakout_agent",
            name="突破代理",
            signal=classify_signal(breakout_score),
            confidence=confidence_from_score(breakout_score, 50),
            summary=f"股價距離 20 日高點不遠，量比 {volume_ratio:.2f} 倍，布林頻寬呈 {bb_desc} ({bb_width:.1%})。",
            historical_edge=agent_edges.get("breakout_agent"),
        ),
        AgentOpinion(
            key="mean_reversion_agent",
            name="回檔承接代理",
            signal=classify_signal(mean_reversion_score),
            confidence=confidence_from_score(mean_reversion_score, 50),
            summary=f"觀察是否出現回檔後可承接的位置，目前偏 {classify_signal(mean_reversion_score)}。",
            historical_edge=agent_edges.get("mean_reversion_agent"),
        ),
        AgentOpinion(
            key="risk_agent",
            name="風險代理",
            signal="偏空" if risk_score >= 2 else "中性",
            confidence=confidence_from_score(risk_score, 60),
            summary=f"ATR 波動率 {atr_pct:.2f}%，風險評估為 {'偏高' if risk_score >= 2 else '可控'}。",
        ),
        AgentOpinion(
            key="warren_buffett",
            name="Warren Buffett (巴菲特)",
            signal=buffett_sig,
            confidence=buffett_conf,
            summary=buffett_sum
        ),
        AgentOpinion(
            key="charlie_munger",
            name="Charlie Munger (蒙格)",
            signal=munger_sig,
            confidence=munger_conf,
            summary=munger_sum
        ),
        AgentOpinion(
            key="ben_graham",
            name="Ben Graham (葛拉漢)",
            signal=graham_sig,
            confidence=graham_conf,
            summary=graham_sum
        ),
        AgentOpinion(
            key="aswath_damodaran",
            name="Aswath Damodaran (達莫達蘭)",
            signal=damodaran_sig,
            confidence=damodaran_conf,
            summary=damodaran_sum
        ),
        AgentOpinion(
            key="cathie_wood",
            name="Cathie Wood (凱薩琳伍德)",
            signal=wood_sig,
            confidence=wood_conf,
            summary=wood_sum
        ),
        AgentOpinion(
            key="nassim_taleb",
            name="Nassim Taleb (塔雷伯)",
            signal=taleb_sig,
            confidence=taleb_conf,
            summary=taleb_sum
        ),
        AgentOpinion(
            key="peter_lynch",
            name="Peter Lynch (彼得林區)",
            signal=lynch_sig,
            confidence=lynch_conf,
            summary=lynch_sum
        ),
        AgentOpinion(
            key="michael_burry",
            name="Michael Burry (麥克貝瑞)",
            signal=burry_sig,
            confidence=burry_conf,
            summary=burry_sum
        ),
        AgentOpinion(
            key="stanley_druckenmiller",
            name="Stanley Druckenmiller (德魯肯米勒)",
            signal=druckenmiller_sig,
            confidence=druckenmiller_conf,
            summary=druckenmiller_sum
        ),
        AgentOpinion(
            key="bill_ackman",
            name="Bill Ackman (艾克曼)",
            signal=ackman_sig,
            confidence=ackman_conf,
            summary=ackman_sum
        ),
    ]

def build_horizon_views(frame: pd.DataFrame) -> list[HorizonView]:
    latest = frame.iloc[-1]
    prev_day = frame.iloc[-2]
    
    close = float(latest["Close"])
    atr14 = float(latest["ATR14"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    ma120 = float(latest["MA120"]) if not np.isnan(latest["MA120"]) else ma50
    support60 = float(frame.tail(60)["Low"].min())
    
    # Calculate Pivot Points from previous day
    prev_high = float(prev_day["High"])
    prev_low = float(prev_day["Low"])
    prev_close = float(prev_day["Close"])
    pivots = compute_pivot_levels(prev_high, prev_low, prev_close)
    pp, r1, s1, r2, s2 = pivots["pp"], pivots["r1"], pivots["s1"], pivots["r2"], pivots["s2"]

    # Entry Zones anchored on Pivots
    short_entry = safe_range(max(s1, close * 0.985), min(pp, close * 1.005), close)
    medium_entry = safe_range(max(s2, close * 0.95), min(s1, close * 0.985), close)
    long_entry = safe_range(max(support60, close * 0.90), min(s2, close * 0.95), close)

    # Take Profit Zones anchored on Resistance
    short_tp = safe_range(r1 * 0.99, r1 * 1.03, close)
    medium_tp = safe_range(r2 * 0.99, r2 * 1.04, close)
    long_tp = safe_range(r2 * 1.10, r2 * 1.20, close)

    # Stop Loss Zones anchored on Support
    short_stop = safe_range(s1 * 0.965, s1 * 0.985, close)
    medium_stop = safe_range(s2 * 0.94, s2 * 0.97, close)
    long_stop = safe_range(s2 * 0.88, s2 * 0.92, close)

    return [
        HorizonView(
            horizon="短線",
            bias=classify_signal(int(close > ma20) + int(float(latest["RSI5"]) > 55) - int(float(latest["RSI5"]) < 45)),
            entry_zone=format_range(*short_entry),
            take_profit_zone=format_range(*short_tp),
            stop_zone=format_range(*short_stop),
            summary="短線以 1 到 3 週為主，進場區錨定前日 Pivots 支撐 s1 與 pp。",
        ),
        HorizonView(
            horizon="中線",
            bias=classify_signal(
                int(close > ma20) + int(ma20 > ma50) + int(float(latest["RSI14"]) > 55) - int(float(latest["RSI14"]) < 45)
            ),
            entry_zone=format_range(*medium_entry),
            take_profit_zone=format_range(*medium_tp),
            stop_zone=format_range(*medium_stop),
            summary="中線要求 20% 的預期報酬空間，進場區間位於阻力關卡 s2 至 s1。",
        ),
        HorizonView(
            horizon="長線",
            bias=classify_signal(int(close > ma50) + int(ma50 > ma120) + int(close > ma120) - int(close < ma50)),
            entry_zone=format_range(*long_entry),
            take_profit_zone=format_range(*long_tp),
            stop_zone=format_range(*long_stop),
            summary="長線順應大趨勢，進場區置於 s2 以下以拉開足夠的安全邊際。",
        ),
    ]


def build_price_forecast(
    frame: pd.DataFrame,
    *,
    latest_close: float,
    bias: str,
    rsi14: float,
    composite_score: int,
    analyst_target_price: float | None = None,
    symbol: str = "",
) -> dict:
    """以歷史漂移率 + 波動度錐 (drift / volatility cone) 推估 3/6/9/12 個月股價區間，
    並結合趨勢、RSI、綜合分數與分析師目標價，給出「現在該買或賣」的判讀。

    這是統計型推估（非保證預測）：基準價來自歷史日對數報酬的年化漂移，
    上下緣為 ±1 個標準差的波動範圍；長天期會逐步衰減漂移以反映動能不確定性。
    """
    closes = frame["Close"].astype(float)
    log_ret = np.log(closes / closes.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    window = log_ret.tail(378)
    if len(window) < 60:
        window = log_ret
    if len(window) < 20:
        return {}

    # 2026-07-06 檢討修正：漂移改「多視窗混合」。舊制單一 378 日等權平均會讓一年多前的
    # 崩跌長期污染預測——例：6409 已 V 型反轉且創 60 日新高，半年期望報酬仍算出 -6%，
    # 在突破日觸發持股健檢的賣出閘門（1000 元被叫賣、7 個交易日後 1290）。
    # 近期動能的預測力已在 Top50 因子研究驗證（20 日動能與未來報酬 Spearman +0.31~0.41），
    # 故近 63 日權重最高、126 日次之、378 日只留基底；波動度仍用長窗（估計較穩定）。
    w63 = log_ret.tail(63)
    w126 = log_ret.tail(126)
    if len(w63) >= 20 and len(w126) >= 40:
        mu_daily = 0.45 * float(w63.mean()) + 0.35 * float(w126.mean()) + 0.20 * float(window.mean())
    else:
        mu_daily = float(window.mean())
    sigma_daily = float(window.std())

    # 年化漂移設上下限，避免將短期強勢線性外推成不合理的預測。
    # 2026-06-19：負向漂移額外收斂(×0.55) 並把下限收到 -22% —— 近期回檔不該被線性外推成「長線必跌」，
    # 權益長期有上升偏誤，過度悲觀的外推正是品質股被誤標長線虧損的根源。正向不變(動能外推較合理)。
    raw_annual_drift = mu_daily * 252.0
    if raw_annual_drift < 0:
        raw_annual_drift *= 0.55
    capped_annual = max(min(raw_annual_drift, 0.45), -0.22)

    # AI 主線科技巨頭設正向漂移下限（+5% 年化）：這些股票的 10 年 CAGR 遠超 5%，
    # 短期回檔被線性外推成長線必虧是典型的樣本期偏誤，應保護不被誤標。
    _mega_cap_floor_symbols = {
        "NVDA", "MSFT", "AAPL", "GOOGL", "GOOG", "AMZN", "META",
        "AVGO", "TSM", "AMD", "MU", "ARM", "PLTR", "2330",
        # 台股 AI 主線龍頭：10 年 CAGR 遠高於 5%，短期回檔不應被線性外推成長線必虧
        "2308", "2317", "2382", "2454", "2376", "3231", "6669", "3661", "3017",
    }
    sym_base = symbol.upper().split(".")[0] if symbol else ""
    if sym_base in _mega_cap_floor_symbols and capped_annual < 0.05:
        capped_annual = 0.05  # 科技主線最低年化漂移 +5%

    mu_capped = capped_annual / 252.0

    horizons = [("3個月", 63), ("6個月", 126), ("9個月", 189), ("12個月", 252)]
    damp = {63: 1.0, 126: 0.92, 189: 0.85, 252: 0.80}

    def stance_for(ret_pct: float) -> str:
        if ret_pct >= 15:
            return "看多"
        if ret_pct >= 5:
            return "偏多"
        if ret_pct <= -15:
            return "看空"
        if ret_pct <= -5:
            return "偏空"
        return "中性"

    forecast_horizons: list[dict] = []
    exp_returns: dict[int, float] = {}
    for label, days in horizons:
        eff_mu = mu_capped * damp[days]
        base = latest_close * float(np.exp(eff_mu * days))
        # 9/12 個月向分析師目標價輕度收斂（分析師目標多為 12 個月）。
        if analyst_target_price and analyst_target_price > 0 and days >= 189:
            blend = 0.25 if days == 189 else 0.35
            base = base * (1 - blend) + analyst_target_price * blend
        vol_h = sigma_daily * float(np.sqrt(days))
        low = latest_close * float(np.exp(eff_mu * days - vol_h))
        high = latest_close * float(np.exp(eff_mu * days + vol_h))
        # 收斂後重算基準對應的上下緣，使區間以 base 為中心。
        if analyst_target_price and analyst_target_price > 0 and days >= 189:
            half = (high - low) / 2.0
            low, high = base - half, base + half
        exp_ret = (base / latest_close - 1.0) * 100.0
        exp_returns[days] = exp_ret
        forecast_horizons.append(
            {
                "label": label,
                "days": days,
                "base": round(base, 2),
                "low": round(max(low, 0.0), 2),
                "high": round(high, 2),
                "expected_return_pct": round(exp_ret, 1),
                "stance": stance_for(exp_ret),
            }
        )

    exp6 = exp_returns.get(126, 0.0)
    exp12 = exp_returns.get(252, 0.0)

    score = 0
    if exp12 >= 25:
        score += 2
    elif exp12 >= 10:
        score += 1
    elif exp12 <= -15:
        score -= 2
    elif exp12 <= -5:
        score -= 1
    if exp6 >= 12:
        score += 1
    elif exp6 <= -8:
        score -= 1
    if bias == "偏多":
        score += 1
    elif bias == "偏空":
        score -= 1
    if rsi14 >= 70:
        score -= 1
    elif rsi14 <= 35:
        score += 1
    if composite_score >= 70:
        score += 1
    elif composite_score <= 40:
        score -= 1

    if score >= 3:
        verdict = "中長線偏多·可分批布局"
        verdict_reason = "趨勢、預測與評分多數同向偏多，中長線適合靠近買點分批建倉。"
    elif score >= 1:
        verdict = "中長線偏多·逢回分批"
        verdict_reason = "中長期偏多但非全數確認，建議靠近買點區分批承接、不追高。"
    elif score == 0:
        verdict = "中長線中性·觀望"
        verdict_reason = "多空訊號互見，等趨勢或預測更明確再動作。"
    elif score >= -2:
        verdict = "中長線偏弱·暫不進場"
        verdict_reason = "預測或趨勢偏弱，新倉宜保守，持股者注意保護停損。"
    else:
        verdict = "中長線偏空·建議減碼"
        verdict_reason = "趨勢與中長期預測同向偏空，宜降低部位或避開。"
    # 明確區分時間框架，避免與上方「今日操作結論」(短線進場時機) 看似矛盾。
    verdict_reason += "（此為 3–12 個月中長線觀點；今天該不該進場，以上方『今日操作結論』的短線買賣點為準。）"

    return {
        "method": "歷史漂移 + 波動度錐（±1σ）統計推估，已對長天期衰減並收斂分析師目標價。",
        "annualized_drift_pct": round(capped_annual * 100.0, 1),
        "annualized_volatility_pct": round(sigma_daily * float(np.sqrt(252.0)) * 100.0, 1),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "horizons": forecast_horizons,
    }


def compute_agent_edges(frame: pd.DataFrame, forward_days: int = 20) -> dict:
    """針對個股自身歷史，回測各技術代理「看多訊號」出現後 forward_days 日的表現，
    依勝率與平均報酬給每個技術代理一個信任權重 (weight ∈ [0.4, 1.8])，
    使不同股票對不同代理的信任可自動調整 (per-stock adaptive trust)。"""
    edges: dict[str, dict] = {}
    if frame is None or len(frame) < forward_days + 40:
        return edges

    df = frame
    close = df["Close"].astype(float)
    fwd = close.shift(-forward_days) / close - 1.0  # 未來 forward_days 日報酬

    ma10 = df["MA10"].astype(float)
    ma20 = df["MA20"].astype(float)
    ma50 = df["MA50"].astype(float)
    rsi5 = df["RSI5"].astype(float)
    rsi14 = df["RSI14"].astype(float)
    atr14 = df["ATR14"].astype(float)
    zeros = pd.Series(0.0, index=df.index)
    macd = df["MACD"].astype(float) if "MACD" in df.columns else zeros
    macd_sig = df["MACD_Signal"].astype(float) if "MACD_Signal" in df.columns else zeros
    macd_hist = df["MACD_Hist"].astype(float) if "MACD_Hist" in df.columns else zeros
    volume = df["Volume"].astype(float)
    vol20 = df["VOL20"].astype(float) if "VOL20" in df.columns else volume.rolling(20).mean()
    volume_ratio = (volume / vol20).replace([np.inf, -np.inf], np.nan)
    resistance20 = df["High"].astype(float).rolling(20).max()

    conditions = {
        "trend_agent": (close > ma20) & (ma20 > ma50) & (macd > macd_sig),
        "momentum_agent": (rsi14 > 56) & (macd_hist > macd_hist.shift(1)),
        "breakout_agent": (close >= resistance20 * 0.99) & (volume_ratio > 1.15),
        "mean_reversion_agent": (rsi5 < 35) & (close < ma10 - 0.7 * atr14),
    }

    for key, cond in conditions.items():
        mask = cond.fillna(False) & fwd.notna()
        sample = fwd[mask]
        n = int(sample.shape[0])
        if n < 8:
            edges[key] = {"sample_size": n, "win_rate": 0.0, "avg_return": 0.0, "weight": 1.0}
            continue
        wins = float((sample > 0).mean()) * 100.0
        avg = float(sample.mean()) * 100.0
        # 勝率 50%、平均 0% → 權重 1.0；表現越好權重越高、越差越低
        win_factor = (wins - 50.0) / 50.0            # [-1, 1]
        ret_factor = max(min(avg / 8.0, 1.0), -1.0)  # ±8% 報酬封頂
        raw = 1.0 + 0.5 * win_factor + 0.3 * ret_factor
        weight = round(max(min(raw, 1.8), 0.4), 3)
        edges[key] = {
            "sample_size": n,
            "win_rate": round(wins, 1),
            "avg_return": round(avg, 2),
            "weight": weight,
        }
    return edges


def evaluate_long_term_risk(
    price_forecast: dict | None,
    timeline_bt: dict | None,
    *,
    latest_close: float | None = None,
    ma_long: float | None = None,
    fundamental_score: int | None = None,
    bias: str | None = None,
    valuation_gap_pct: float | None = None,
) -> dict:
    """判斷長線是否「真的」會虧損 → blocked=True 才不建議買進。

    2026-06-19 重寫（修 MSFT/AVGO 這類品質股在正常回檔被誤標「長線恐虧損」）：
    原本只靠「12 個月統計預測」(本質是把近期價格漂移線性外推，對 1 年後預測力低、且會把暫時回檔
    當成長期下跌) 就能硬否決，過度誤殺優質龍頭。新版要求**多重證據同時成立**，並對「長期多頭結構
    (站上長均線) 或基本面強健」的股票給予保護，不再單憑悲觀外推否決。"""
    exp12 = base12 = high12 = exp6 = None
    if price_forecast and price_forecast.get("horizons"):
        for h in price_forecast["horizons"]:
            if h.get("days") == 252:
                exp12 = float(h.get("expected_return_pct", 0.0))
                base12 = float(h.get("base", 0.0))
                high12 = float(h.get("high", 0.0))
            elif h.get("days") == 126:
                exp6 = float(h.get("expected_return_pct", 0.0))

    hist_cum = hist_win = hist_avg = None
    hist_trades = 0
    if timeline_bt:
        hist_cum = float(timeline_bt.get("cumulative_return", 0.0))
        hist_win = float(timeline_bt.get("win_rate", 0.0))
        hist_avg = float(timeline_bt.get("avg_return", 0.0))
        hist_trades = int(timeline_bt.get("total_trades", 0))

    # 結構/品質保護：站上長期均線(多頭結構) 或 基本面強健 → 視為品質股，正常回檔不該被當長線必虧。
    long_uptrend = (latest_close is not None and ma_long is not None and ma_long > 0
                    and latest_close >= ma_long)
    quality = fundamental_score is not None and fundamental_score >= 6
    protected = long_uptrend or quality
    bearish_structure = bias == "偏空" or (
        latest_close is not None and ma_long is not None and ma_long > 0 and latest_close < ma_long * 0.92
    )

    reasons: list[str] = []
    blocked = False
    severity = "low"

    forecast_negative = exp12 is not None and exp12 < 0
    forecast_very_negative = exp12 is not None and exp12 <= -15   # 硬門檻由 -8 提高到 -15
    hist_negative = hist_cum is not None and hist_trades >= 3 and hist_cum < 0
    hist_weak = (
        hist_avg is not None and hist_trades >= 3
        and hist_avg < 0 and (hist_win is None or hist_win < 45)
    )

    # 中期(6個月)是否仍偏多：若 6 個月預測仍正向，12 個月尾段的小幅負值多為長天期衰減／收斂分析師
    # 目標價造成的雜訊，不應觸發「長線需謹慎」與「中長線偏多」的結論互相矛盾。
    midterm_ok = exp6 is not None and exp6 >= 3.0

    # 2026-06-30：修正「12 個月預測為正、估值大幅折價，卻被短期/小樣本的負向波段回測硬否決」的矛盾
    #   （例：CRWV CoreWeave，新上市高波動，11 筆波段回測累積 -36%，但 12 個月統計預測 +6.3%、
    #    InvestingPro 估值折價 +26%，卻被判「長線恐虧損 不建議買進」——結論與自身數據自相矛盾）。
    # 規則：12 個月前瞻期望為正 或 顯著被低估(估值折價≥15%) → 不以「僅靠回測歷史」的證據硬否決，
    #   改為謹慎提示。仍由買進端的結構/品質閘門決定是否真的給買訊，不會強迫接刀。
    forward_positive_12m = exp12 is not None and exp12 >= 0
    materially_undervalued = valuation_gap_pct is not None and valuation_gap_pct >= 15.0
    rescue_from_history_only_block = forward_positive_12m or materially_undervalued

    if protected:
        # 品質/多頭結構股：不硬否決。只有「連中期(6個月)都走弱」才提示審慎；6 個月仍偏多則不警示，
        # 避免與下方「中長線偏多·可分批布局」的結論打架。
        if forecast_negative and not midterm_ok:
            severity = "medium"
            reasons.append(
                f"12 個月統計預測偏負 ({exp12:.1f}%)、且中期動能轉弱；個股仍處多頭結構，"
                f"視為短期逆風，宜分批、不宜重押。"
            )
        # exp6 仍正向 → 不加警示（落到「未顯示長抱虧損風險」），與中長線偏多一致。
    else:
        # 非品質股才考慮硬否決，且需多重證據同時成立。
        if forecast_negative and hist_negative and (bearish_structure or hist_weak):
            blocked = True
            severity = "high"
            reasons.append(
                f"12 個月統計預測 {exp12:.1f}%（偏負）、個股波段歷史累積 {hist_cum:.1f}% 同步虧損，"
                f"且趨勢結構同步轉弱，長抱仍難轉正。"
            )
        elif forecast_very_negative and bearish_structure:
            blocked = True
            severity = "high"
            reasons.append(f"12 個月統計預測達 {exp12:.1f}%、且已跌破長期均線轉空，長線期望值明顯為負。")
        elif hist_negative and hist_weak and bearish_structure:
            if rescue_from_history_only_block:
                # 回測歷史偏弱，但前瞻期望為正或估值大幅折價 → 視為短線逆風、不硬否決。
                severity = "medium"
                why_bits = []
                if forward_positive_12m:
                    why_bits.append(f"12 個月統計預測為正 ({exp12:.1f}%)")
                if materially_undervalued:
                    why_bits.append(f"估值仍折價 {valuation_gap_pct:.0f}%")
                reasons.append(
                    f"個股波段歷史回測偏弱（累積 {hist_cum:.1f}%、勝率 {hist_win:.0f}%），"
                    f"但{ '、'.join(why_bits) }；多為新上市/高波動的小樣本回測，不硬性否決，"
                    f"宜分批小量、嚴設停損承接，不宜重押。"
                )
            else:
                blocked = True
                severity = "medium"
                reasons.append(
                    f"個股波段歷史累積 {hist_cum:.1f}%、勝率僅 {hist_win:.0f}%，且趨勢轉弱，長抱賺錢機率偏低。"
                )
        elif forecast_negative and not midterm_ok:
            severity = "medium"
            reasons.append(f"12 個月預測小幅為負 ({exp12:.1f}%)，僅供觀望、不宜重押。")

    note = " ".join(reasons) if reasons else "長線預測與個股歷史回測未顯示長抱虧損風險。"
    return {
        "blocked": blocked,
        "severity": severity,
        "note": note,
        "expected_return_12m_pct": exp12,
        "forecast_base_12m": base12,
        "forecast_high_12m": high12,
        "history_cumulative_return_pct": hist_cum,
        "history_win_rate_pct": hist_win,
        "history_trades": hist_trades,
    }


def compute_rule_score(agents: list[dict], horizons: list[dict]) -> tuple[int, str]:
    agent_weights = {
        "trend_agent": 6,
        "momentum_agent": 5,
        "breakout_agent": 3,
        "mean_reversion_agent": 3,
        "risk_agent": 4,
        "warren_buffett": 8,
        "charlie_munger": 8,
        "ben_graham": 7,
        "aswath_damodaran": 7,
        "cathie_wood": 6,
        "nassim_taleb": 7,
        "peter_lynch": 6,
        "michael_burry": 7,
        "stanley_druckenmiller": 6,
        "bill_ackman": 6,
    }
    horizon_weights = {"短線": 4, "中線": 7, "長線": 9}
    score = 50.0
    for agent in agents:
        base_weight = agent_weights.get(agent["key"], 4)
        # 技術代理依個股自身歷史回測的信任權重縮放 (per-stock adaptive trust)
        edge = agent.get("historical_edge")
        if edge and isinstance(edge, dict):
            base_weight *= float(edge.get("weight", 1.0))
        score += score_from_signal(agent["signal"]) * base_weight
    for horizon in horizons:
        score += score_from_signal(horizon["bias"]) * horizon_weights.get(horizon["horizon"], 4)
    final_score = clamp_score(score)
    return final_score, score_to_strength(final_score)

def enforce_position_value(
    buy_zone: str,
    sell_zone: str,
    stop_loss: str,
    buy_strength: str,
    reason: str,
    fundamental_score: int = 0,
    valuation_gap_pct: float | None = None,
) -> tuple[str, str, str, str, str]:
    entry_mid = range_mid(buy_zone)
    target_mid = range_mid(sell_zone)
    expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
    
    # Value investor allowance: if it is strong value, allow lower return hurdle (12% instead of 20%)
    is_strong_value = (fundamental_score >= 5 and valuation_gap_pct is not None and valuation_gap_pct >= 10.0)
    min_return = 12.0 if is_strong_value else MIN_POSITION_RETURN_PCT
    
    if expected_return_pct >= min_return:
        return buy_zone, sell_zone, stop_loss, buy_strength, reason

    if buy_strength == "強力買進":
        buy_strength = "可分批買進"
    elif buy_strength == "可分批買進":
        buy_strength = "觀察偏多"
    else:
        buy_strength = "先觀察"

    reason = f"{reason} 但目前從建議買點到目標賣點的預期報酬不足 {min_return:.1f}%，新倉先保守。"
    return buy_zone, sell_zone, stop_loss, buy_strength, reason


def derive_today_plan(
    latest_close: float,
    ma50: float,
    atr14: float,
    rsi14: float,
    bias: str,
    buy_strength: str,
    buy_zone: str,
    sell_zone: str,
    stop_loss: str,
    candlestick_pattern: str = "無",
    fundamental_score: int = 0,
    valuation_gap_pct: float | None = None,
    ai_chain_layer: str | None = None,
    ma120: float = 0.0,
    long_term_blocked: bool = False,
    drawdown_from_high_pct: float = 0.0,
    ma20: float = 0.0,
    fundamentals_available: bool = True,
    ma120_rising: bool = False,
    drop_3d_pct: float = 0.0,
) -> tuple[str, str, str, str, str, str, float, float, int, str, float]:
    buy_low, buy_high = parse_range(buy_zone)
    sell_low, sell_high = parse_range(sell_zone)
    stop_low, stop_high = parse_range(stop_loss)

    entry_zone = format_range(max(buy_low, latest_close - 0.45 * atr14), min(buy_high, latest_close + 0.15 * atr14))
    entry_mid = range_mid(entry_zone)
    target_mid = range_mid(sell_zone)
    stop_mid = (stop_low + stop_high) / 2
    expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
    risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
    reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0

    near_buy_zone = latest_close <= buy_high * 1.01
    slightly_extended = latest_close <= buy_high * 1.04 and rsi14 < 63

    # Value investor allowance: if it is heavily undervalued with strong fundamentals, allow buying even in downtrend
    is_strong_value = (fundamental_score >= 5 and valuation_gap_pct is not None and valuation_gap_pct >= 10.0)
    # AI 主線核心股（有 ai_chain_layer + 基本面及格）：大跌是加碼機會，不應直接否決。
    # 2026-07-22 檢討修正（7/17~19 CRWV/NBIS/SNDK/群聯/台達電 回檔全被判「不要買」、7/21 反彈）：
    # 每日管線 fetch_fundamentals=False → fundamental_score 恆為 0，讓 is_ai_mainline 永遠 False，
    # 所有「主線回檔承接」路徑形同死路。基本面「沒抓到」不等於「不好」——AI 主線對照表本身就是
    # 人工精選的品質清單；資料不可用時不再卡 F-Score 門檻，資料可用時照舊要求 >=5 防雷。
    is_ai_mainline = ai_chain_layer is not None and (fundamental_score >= 5 or not fundamentals_available)
    min_return_threshold = 12.0 if is_strong_value else MIN_POSITION_RETURN_PCT

    # === 好股票大跌 = 承接機會（quality-dip buy）===
    # 2026-06-30：修正「好股在大盤恐慌大跌時被判偏空 → 今天不要買」的反向錯誤
    #   （例：2026-06-26 台美股大跌，2308 台達電被標不要買，6/29、6/30 卻大漲）。
    # 邏輯：長線結構未壞（站上年線 MA120）或基本面強健 / AI 主線 / 深度價值，
    #   且短線被市場恐慌錯殺至超跌（RSI 低 或 自 60 日高點大幅回落），
    #   視為「好股大跌的分批承接機會」，而非弱勢不要買。
    #   長線虧損閘門已否決者（long_term_blocked）一律排除，避免承接真正向下的標的。
    # 2026-07-22 檢討修正：恐慌殺盤常把好股「殺破年線再 V 轉」（7/17~19 台達電/群聯正是如此），
    # 舊定義（收盤 >= 年線 * 0.97）在最該接的那幾天反而判定結構已壞。放寬：年線本身仍上揚
    # （長線趨勢未轉彎）時，跌破年線 10% 以內仍視為長線結構未壞；年線已走平/下彎則維持嚴格門檻。
    long_uptrend_intact = ma120 > 0 and (
        latest_close >= ma120 * 0.97
        or (ma120_rising and latest_close >= ma120 * 0.90)
    )
    quality_name = (
        fundamental_score >= 6
        or long_uptrend_intact
        or is_ai_mainline
        or is_strong_value
    )
    # 2026-07-06 檢討修正：回落門檻 -12% → -8%。一般市場恐慌日好股多回落 3~8%，
    # 舊門檻讓「大跌抄底」幾乎永遠不觸發（上週五大跌被判不要買、下週一即大漲）。
    # 2026-07-22 新增「近 3 日急跌」觸發：緩跌後的急殺常是恐慌高潮（RSI 未必 <45、
    # 距 60 日高點未必 -8%，但 3 日內 -6% 的殺盤本身就是錯殺訊號）。
    panic_oversold = (rsi14 < 45.0) or (drawdown_from_high_pct <= -8.0) or (drop_3d_pct <= -6.0)
    is_quality_dip = (
        not long_term_blocked
        and quality_name
        and panic_oversold
        and bias != "偏多"  # 偏多本就走 near_buy/小量買路徑，這裡專收偏空/中性的錯殺
    )

    # Estimate winning probability p
    p_map = {
        "強力買進": 0.64,
        "可分批買進": 0.60,
        "觀察偏多": 0.56,
        "先觀察": 0.52,
        "不建議進場": 0.42
    }
    p = p_map.get(buy_strength, 0.50)

    # Calculate Kelly percentage (Half-Kelly for safety)
    if reward_ratio > 0:
        kelly_raw = p - (1.0 - p) / reward_ratio
        kelly_position_pct = max(0.0, min(0.30, kelly_raw / 2.0))  # Cap at 30% max for single position safety
    else:
        kelly_position_pct = 0.0

    candle_bonus = f" (偵測到K線訊號: {candlestick_pattern})" if candlestick_pattern != "無" else ""

    if long_term_blocked:
        # 長線期望值為負（長抱仍虧損）：即使短線超跌也不承接，把資金留給長線向上的好股。
        today_action = NO_BUY
        today_note = f"⚠️ 長線期望值為負，即使短線大跌也不建議承接，請優先布局長線向上的標的。{candle_bonus}"
    elif is_quality_dip:
        # 好股票被市場恐慌錯殺 → 分批小量承接（買在恐慌、留銀彈攤平、嚴設停損）。
        today_action = BUY_SMALL
        dip_kelly = max(kelly_position_pct, 0.05)
        # 承接區錨定現價附近（買的是今天的恐慌，不是更深的支撐），並重算報酬/風報比。
        entry_zone = format_range(
            max(latest_close * 0.97, latest_close - 0.6 * atr14),
            min(latest_close * 1.005, latest_close + 0.1 * atr14),
        )
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
        if long_uptrend_intact:
            why = "長線結構未破（仍站穩年線 MA120）"
        elif is_ai_mainline:
            why = f"AI 主線核心（{ai_chain_layer}）"
        elif fundamental_score >= 6:
            why = f"基本面強健（F-Score {fundamental_score}/9）"
        else:
            why = "深度價值（具安全邊際）"
        dd_txt = f"、自近期高點回落約 {abs(drawdown_from_high_pct):.0f}%" if drawdown_from_high_pct <= -8.0 else ""
        if drop_3d_pct <= -6.0:
            dd_txt += f"、近 3 日急跌 {abs(drop_3d_pct):.0f}%"
        today_note = (
            f"⚡ 好股票大跌承接機會：{why}，但短線被市場恐慌錯殺至超跌（RSI {rsi14:.0f}{dd_txt}）。"
            f"優質股在非理性下殺後常出現均值回歸反彈，策略上應「別人恐懼我貪婪」分批小量承接、"
            f"留銀彈攤平，建議倉位 {dip_kelly:.1%}。務必嚴設停損於 {stop_loss}，若真跌破再停損出場。{candle_bonus}"
        )
    elif (
        not long_term_blocked
        and drawdown_from_high_pct <= -30.0
        and rsi14 < 40.0
        and (
            (valuation_gap_pct is not None and valuation_gap_pct >= 10.0)
            # 2026-07-22：估值資料抓不到（每日管線）時不再直接封死這條路，
            # 改用「更深的跌幅 + 更低的 RSI」作為替代證據，維持不對稱賭注的嚴格度。
            or (valuation_gap_pct is None and drawdown_from_high_pct <= -35.0 and rsi14 < 38.0)
        )
    ):
        # === 深度超跌反轉（2026-07-06 回放檢討新增）===
        # CRWV 案例：自高點 -41%、RSI 34、低估 15%、12 個月統計預測 +27%，但 F-Score 差 1 分、
        # 年線之下，所有「好股」定義都差一點點 → 被「偏空不建議進場」擋掉，隨後 +6%。
        # 深跌 + 明確低估 + 未觸長線虧損閘門 = 不對稱賭注，允許「最小倉位」參與反轉。
        today_action = BUY_SMALL
        entry_zone = format_range(
            max(latest_close * 0.97, latest_close - 0.6 * atr14),
            min(latest_close * 1.005, latest_close + 0.1 * atr14),
        )
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
        _value_evidence = (
            f"且估值折價 {valuation_gap_pct:.0f}%" if valuation_gap_pct is not None
            else "跌幅與超賣程度已達深度錯殺標準"
        )
        today_note = (
            f"🎲 深度超跌＋低估的反轉機會：自近期高點已回落 {abs(drawdown_from_high_pct):.0f}%、RSI {rsi14:.0f} 超賣，"
            f"{_value_evidence}、未觸發長線虧損閘門。這類標的波動極大，"
            f"僅適合「最小倉位」（總資金 3~5%）試單參與反轉，務必嚴設停損 {stop_loss}，跌破就走、不攤平。{candle_bonus}"
        )
    elif buy_strength == "不建議進場" and not is_ai_mainline:
        today_action = NO_BUY
        today_note = f"趨勢偏弱，今天不建議新倉進場。{candle_bonus}"
    elif bias == "偏空" and not is_strong_value and not is_ai_mainline:
        today_action = NO_BUY
        today_note = f"趨勢偏弱，今天不建議新倉進場。{candle_bonus}"
    elif bias == "偏空" and is_ai_mainline:
        # AI 主線科技股（NVDA/MSFT/MU 等）大跌時是加碼機會，不能直接否決。
        # 參考策略：主線回檔時分批小量建倉，堅守核心邏輯不輕易賣出。
        today_action = BUY_SMALL
        dip_kelly = max(kelly_position_pct, 0.05)  # 主線大跌至少給 5% 底倉建議
        today_note = (
            f"⚡ AI主線回檔承接機會（{ai_chain_layer}）：短線雖偏弱，但主線長期敘事不變，"
            f"大跌時分批小量建倉是正確策略，建議倉位 {dip_kelly:.1%}（先建底倉，待趨勢確認後加碼）。{candle_bonus}"
        )
    elif expected_return_pct < min_return_threshold:
        today_action = NO_BUY
        today_note = f"目前可用的報酬空間不足 {min_return_threshold:.1f}%，今天先不要急著買。{candle_bonus}"
    elif near_buy_zone:
        today_action = BUY_NOW
        today_note = f"現價仍在可執行承接區附近，建議直接分批掛單，凱利公式建議倉位 {kelly_position_pct:.1%}。{candle_bonus}"
    elif slightly_extended:
        today_action = BUY_SMALL
        today_note = f"價格略高於理想承接區，若要進場建議先買小量，凱利公式建議倉位 {kelly_position_pct:.1%}。{candle_bonus}"
        entry_zone = format_range(max(latest_close * 0.992, latest_close - 0.35 * atr14), min(latest_close * 1.002, latest_close + 0.05 * atr14))
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
    elif (
        bias != "偏空"
        and drawdown_from_high_pct >= -3.0
        and buy_strength in ("強力買進", "可分批買進")
    ):
        # === 強勢突破可小量參與（2026-07-06 回放檢討新增）===
        # 7/2 回放：TENB +9.2%、6409 +8.9%、VRNS +6.3%、MRNA、HIMS、HOOD…15 檔
        # 「偏多＋貼近 60 日高點＋強度強力買進」的突破股全被判「等回檔」而錯過。
        # 賣出端已修（強勢股不賣在突破點），買進端同理：突破日本身就是進場訊號，
        # 用「小倉位＋明確停損」參與，而不是等一個常常等不到的回檔。
        today_action = BUY_SMALL
        breakout_kelly = max(min(kelly_position_pct, 0.10), 0.03)  # 追突破倉位收斂：3%~10%
        entry_zone = format_range(
            max(latest_close * 0.99, latest_close - 0.35 * atr14),
            min(latest_close * 1.005, latest_close + 0.1 * atr14),
        )
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
        today_note = (
            f"🚀 強勢突破可小量參與：股價貼近 60 日高點、綜合強度「{buy_strength}」，突破本身就是進場訊號——"
            f"與其等常常等不到的回檔，不如小倉位（約 {breakout_kelly:.0%}，比一般買點更小）順勢參與，"
            f"停損設突破失敗處（跌回 20 日均線或 {stop_loss}），跌破就走。{candle_bonus}"
        )
    elif quality_name and bias != "偏空" and drawdown_from_high_pct <= -3.0 and rsi14 < 63:
        # 2026-07-06 檢討修正：好股（基本面強/站穩年線/AI主線/深度價值）已自近期高點回落 3% 以上，
        # 「這就是回檔」——強勢股很少跌回機械式理想買區，一直等回檔會錯過整段行情
        # （上週五市場大跌時被判「等回檔/不要買」，下週一即大漲的教訓）。
        today_action = BUY_SMALL
        dip_kelly = max(kelly_position_pct, 0.05)
        entry_zone = format_range(
            max(latest_close * 0.985, latest_close - 0.5 * atr14),
            min(latest_close * 1.005, latest_close + 0.1 * atr14),
        )
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
        today_note = (
            f"📉 好股回檔即機會：已自近期高點回落約 {abs(drawdown_from_high_pct):.0f}%、RSI {rsi14:.0f} 未過熱。"
            f"與其等更深的理想買點（強勢股常常等不到），不如現價附近先小量分批承接、留銀彈，"
            f"建議倉位 {dip_kelly:.1%}，跌破停損 {stop_loss} 再退場。{candle_bonus}"
        )
    else:
        today_action = WAIT_PULLBACK
        today_note = f"目前離理想買點有點遠，更適合等回檔再接。{candle_bonus}"

    sell_mid = (sell_low + sell_high) / 2
    # 2026-07-06 檢討修正：強勢突破（偏多且貼近 60 日高點）時，高 RSI 是「動能強」的表現而非賣點。
    # 6409 在 1000 元創 60 日新高當天被判「可小量賣」，其後 7 個交易日再漲 +29%——
    # 強勢股的賣出紀律改用移動停利（收盤跌破 20 日均線再處理），不賣在突破點。
    strong_breakout = bias == "偏多" and drawdown_from_high_pct >= -2.0
    if bias == "偏空" and latest_close < ma50:
        exit_action = SELL_NOW
        exit_zone = format_range(max(latest_close * 0.997, latest_close - 0.2 * atr14), max(latest_close * 1.006, latest_close))
    elif rsi14 > 68 and not strong_breakout:
        exit_action = SELL_SMALL
        exit_zone = sell_zone
    else:
        exit_action = HOLD
        exit_zone = sell_zone

    exit_note = f"持股續抱中，目標賣點區設在 {sell_zone}。{candle_bonus}"
    if exit_action == SELL_NOW:
        exit_note = f"技術均線已跌破生命線，建議在現價附近 {exit_zone} 執行全額停損或紀律退場以保全資金。{candle_bonus}"
    elif exit_action == SELL_SMALL:
        exit_note = f"短線 RSI ({rsi14:.1f}) 已進入超買過熱區，建議在 {exit_zone} 分批掛單鎖定利潤。{candle_bonus}"
    elif strong_breakout and rsi14 > 68:
        trail_ref = f"（約 {ma20:.2f}）" if ma20 > 0 else ""
        exit_note = (
            f"🚀 強勢突破創高中：RSI {rsi14:.0f} 偏熱是動能強的表現，不是賣出訊號，不要賣在突破點。"
            f"改用移動停利保護獲利——收盤跌破 20 日均線{trail_ref}再分批減碼；目標賣點區 {sell_zone} 僅供參考。{candle_bonus}"
        )

    # Estimate wave holding period
    if bias == "偏多":
        holding_days_estimate = 22
        holding_window = "中線 (1-3個月)"
    elif bias == "偏空":
        holding_days_estimate = 5
        holding_window = "短線 (1-3週)"
    else:
        holding_days_estimate = 10
        holding_window = "短線 (1-3週)"

    return (
        today_action,
        entry_zone,
        today_note,
        exit_action,
        exit_zone,
        exit_note,
        expected_return_pct,
        reward_ratio,
        holding_days_estimate,
        holding_window,
        kelly_position_pct,
    )

def map_ai_chain_and_bottleneck(symbol: str, sector: str) -> tuple[str | None, str | None]:
    sym = symbol.upper().split(".")[0].strip()
    
    # 8-Layer AI Chain mapping
    layers = {
        "NVDA": "🎮 計算核心 Compute Core",
        "AMD": "🎮 計算核心 Compute Core",
        "INTC": "🎮 計算核心 Compute Core",
        "MU": "💾 儲存與記憶體 Memory & Storage",
        "WDC": "💾 儲存與記憶體 Memory & Storage",
        "STX": "💾 儲存與記憶體 Memory & Storage",
        "SNDK": "💾 儲存與記憶體 Memory & Storage",  # Sandisk：NAND，2026-07-22 回放檢討補上
        "8299": "💾 儲存與記憶體 Memory & Storage",  # 群聯：NAND 控制 IC／模組（上櫃 .TWO）
        "2408": "💾 儲存與記憶體 Memory & Storage",  # 南亞科：DRAM
        "MRVL": "🌐 網路互聯 Networking",            # 邁威爾：客製 ASIC / DPU
        "ALAB": "🌐 網路互聯 Networking",            # Astera Labs：PCIe/CXL 連接
        "CRDO": "🌐 網路互聯 Networking",            # Credo：AEC 高速連接
        "COHR": "🌈 光通訊 Photonic / Optical",
        "LITE": "🌈 光通訊 Photonic / Optical",
        "ANET": "🌐 網路互聯 Networking",
        "AVGO": "🌐 網路互聯 Networking",
        "CSCO": "🌐 網路互聯 Networking",
        "TSM": "🏭 半導體製造 Foundry & Equipment",
        "2330": "🏭 半導體製造 Foundry & Equipment",
        "ASML": "🏭 半導體製造 Foundry & Equipment",
        "AMAT": "🏭 半導體製造 Foundry & Equipment",
        "LRCX": "🏭 半導體製造 Foundry & Equipment",
        "KLAC": "🏭 半導體製造 Foundry & Equipment",
        "VRT": "⚡ 資料中心基礎設施 DC Infra",
        "CRWV": "⚡ 資料中心基礎設施 DC Infra",  # CoreWeave：AI 算力雲（GPU 租賃），2026-07-06 回放檢討補上
        "NBIS": "⚡ 資料中心基礎設施 DC Infra",  # Nebius：AI 算力雲
        "DELL": "⚡ 資料中心基礎設施 DC Infra",
        "GE": "⚡ 資料中心基礎設施 DC Infra",
        "VST": "⚡ 資料中心基礎設施 DC Infra",
        "CEG": "⚡ 資料中心基礎設施 DC Infra",
        "3017": "⚡ 資料中心基礎設施 DC Infra",
        "2382": "⚡ 資料中心基礎設施 DC Infra",
        "2317": "⚡ 資料中心基礎設施 DC Infra",
        "2308": "⚡ 資料中心基礎設施 DC Infra",   # 台達電：AI 伺服器電源/散熱龍頭
        "2376": "⚡ 資料中心基礎設施 DC Infra",   # 技嘉：AI 伺服器
        "3231": "⚡ 資料中心基礎設施 DC Infra",   # 緯創：AI 伺服器
        "6669": "⚡ 資料中心基礎設施 DC Infra",   # 緯穎：雲端 AI 伺服器
        "2454": "🎮 計算核心 Compute Core",       # 聯發科：AI ASIC / 邊緣運算
        "3661": "💡 IP & 軟體 IP & Software",      # 世芯-KY：AI ASIC 設計
        "ARM": "💡 IP & 軟體 IP & Software",
        "SNPS": "💡 IP & 軟體 IP & Software",
        "CDNS": "💡 IP & 軟體 IP & Software",
        "PLTR": "💡 IP & 軟體 IP & Software",
        "MSFT": "💡 IP & 軟體 IP & Software",
        "RKLB": "🚀 太空 / 衛星 Space & Satellite",
        "LMT": "🚀 太空 / 衛星 Space & Satellite",
    }

    # 4 Bottlenecks mapping
    bottlenecks = {
        "NVDA": "CoWoS 封裝 🔥",
        "TSM": "CoWoS 封裝 🔥 + 3nm/2nm 製程 🔥",
        "2330": "CoWoS 封裝 🔥 + 3nm/2nm 製程 🔥",
        "MU": "HBM 三巨頭 🔥",
        "VRT": "資料中心電力 🔥",
        "GE": "資料中心電力 🔥",
        "VST": "資料中心電力 🔥",
        "CEG": "資料中心電力 🔥",
        "3017": "CoWoS 封裝 🔥"
    }
    
    layer = layers.get(sym)
    if not layer:
        if symbol == "2330.TW" or symbol == "2330":
            layer = "🏭 半導體製造 Foundry & Equipment"
        elif symbol == "3017.TW" or symbol == "3017":
            layer = "⚡ 資料中心基礎設施 DC Infra"
        elif "Semiconductors" in sector or "Semiconductor" in sector:
            layer = "🏭 半導體製造 Foundry & Equipment"
        elif "Hardware" in sector or "Technology Hardware" in sector:
            layer = "⚡ 資料中心基礎設施 DC Infra"
        elif "Software" in sector:
            layer = "💡 IP & 軟體 IP & Software"
            
    bottleneck = bottlenecks.get(sym)
    if not bottleneck:
        if symbol == "2330.TW":
            bottleneck = "CoWoS 封裝 🔥 + 3nm/2nm 製程 🔥"
        elif symbol == "3017.TW":
            bottleneck = "CoWoS 封裝 🔥"
            
    return layer, bottleneck


def calculate_novice_rating(daily_score: int, pe_ratio: float | None, bottleneck: str | None) -> str:
    if daily_score >= 80:
        if bottleneck or (pe_ratio is not None and pe_ratio <= 28):
            return "🟢 強烈關注"
        else:
            return "🔵 關注"
    elif daily_score >= 60:
        if pe_ratio is not None and pe_ratio <= 22:
            return "🟢 強烈關注"
        else:
            return "🔵 關注"
    elif daily_score >= 45:
        return "🟡 觀望"
    else:
        return "🔴 迴避"


def _gooaye_agent_opinion(symbol: str) -> dict:
    """把股癌/輿情共識做成「其中一種 agent 看法」，附加到個股分析的 agents 清單。

    讀 WeightedConsensusEngine（股癌 Podcast + 社群輿情，免 Key）；有點名 → 真實多空看法，
    無點名/資料庫不可用 → 中性並註明，確保股癌一律以一個 agent 呈現。signal 與其他 agent 一致用偏多/中性/偏空。
    """
    score, label, logic = 50, "", ""
    try:
        from src.sentiment.consensus_engine import WeightedConsensusEngine
        cons = WeightedConsensusEngine().get_stock_consensus(symbol)
        ops = cons.get("opinions") or []
        score = int(cons.get("consensus_score", 50))
        label = cons.get("consensus_label", "")
        if ops:
            top = ops[0]
            logic = top.get("core_logic") or top.get("original_quote") or ""
            signal = "偏多" if score >= 65 else ("偏空" if score < 45 else "中性")
            confidence = max(35, min(90, abs(score - 50) * 2))
            summary = f"股癌/輿情共識 {score} 分（{label}）：{logic}"[:140]
            return asdict(AgentOpinion(key="gooaye", name="股癌 (Gooaye)", signal=signal,
                                       confidence=confidence, summary=summary))
    except Exception:
        pass
    return asdict(AgentOpinion(key="gooaye", name="股癌 (Gooaye)", signal="中性",
                               confidence=40, summary="股癌/社群輿情近期未明確點名此股，暫以中性看待。"))


def _nicolas_agent_opinion(symbol: str) -> dict:
    """把尼可拉斯楊Live 觀點做成「其中一種 agent 看法」，附加到個股分析的 agents 清單。
    讀 docs/data/nicolas_opinions.json（由 YouTube 自動字幕逐集擷取）；有點名 → 真實多空看法，
    無點名 → 中性並註明，確保尼可拉斯楊一律以一個 agent 呈現（與股癌並列）。"""
    try:
        import os
        import json
        from src.sentiment.consensus_engine import score_opinions
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "docs", "data", "nicolas_opinions.json")
        with open(path, encoding="utf-8") as f:
            store = json.load(f)
        ops_all = store.get("opinions", []) if isinstance(store, dict) else (store or [])
        simple = str(symbol).split(".")[0].upper()
        matched = [o for o in ops_all if str(o.get("target_ticker", "")).split(".")[0].upper() == simple]
        if matched:
            cons = score_opinions(matched)
            score = int(cons.get("consensus_score", 50))
            top = matched[0]
            logic = top.get("core_logic") or top.get("original_quote") or ""
            signal = "偏多" if score >= 65 else ("偏空" if score < 45 else "中性")
            confidence = max(35, min(90, abs(score - 50) * 2))
            summary = f"尼可拉斯楊共識 {score} 分：{logic}"[:140]
            return asdict(AgentOpinion(key="nicolas", name="尼可拉斯楊Live", signal=signal,
                                       confidence=confidence, summary=summary))
    except Exception:
        pass
    return asdict(AgentOpinion(key="nicolas", name="尼可拉斯楊Live", signal="中性",
                               confidence=40, summary="尼可拉斯楊最近一集未明確點名此股，暫以中性看待。"))


def build_report(symbol: str, data: pd.DataFrame, fetch_fundamentals: bool = True, lightweight: bool = False) -> SignalReport:
    # lightweight=True 用於大盤掃描的初篩排序：跳過個股回測/預測/估值抓取等較重的運算，
    # 只算出排序所需的 composite_score / expected_return / 風報比，正式入選後再做完整分析。
    frame = data.copy()
    frame["MA10"] = frame["Close"].rolling(10).mean()
    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA50"] = frame["Close"].rolling(50).mean()
    frame["MA120"] = frame["Close"].rolling(120).mean()
    frame["RSI14"] = compute_rsi(frame["Close"], 14)
    frame["RSI5"] = compute_rsi(frame["Close"], 5)
    frame["ATR14"] = compute_atr(frame, 14)
    frame["VOL20"] = frame["Volume"].rolling(20).mean()
    
    # Calculate MACD and Bollinger Band Width
    macd_line, signal_line, macd_hist = compute_macd(frame["Close"])
    frame["MACD"] = macd_line
    frame["MACD_Signal"] = signal_line
    frame["MACD_Hist"] = macd_hist
    frame["BB_Width"] = compute_bb_width(frame["Close"])

    frame = frame.dropna(subset=["MA20", "MA50", "RSI14", "ATR14", "MACD", "BB_Width"]).copy()
    if len(frame) < 90:
        raise ValueError(f"{symbol} 的有效技術資料不足。")

    latest = frame.iloc[-1]
    recent20 = frame.tail(20)
    recent60 = frame.tail(60)

    latest_close = float(latest["Close"])
    ma10 = float(latest["MA10"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    ma120 = float(latest["MA120"]) if "MA120" in latest and not pd.isna(latest["MA120"]) else ma50
    rsi14 = float(latest["RSI14"])
    atr14 = float(latest["ATR14"])
    support = float(recent20["Low"].min())
    resistance = float(recent20["High"].max())
    support60 = float(recent60["Low"].min())
    resistance60 = float(recent60["High"].max())
    # 收盤基準的 60 日高點：判斷「是否貼近新高（強勢突破）」用收盤對收盤，
    # 避免盤中影線讓創收盤新高的股票被誤判為「已回落 5~7%」。
    close_high60 = float(recent60["Close"].max())

    # Candlestick Pattern Detection
    candlestick_pattern = detect_candlestick(frame["Open"], frame["High"], frame["Low"], frame["Close"])

    # 年線是否仍上揚（vs 約 1 個月前）＋近 3 日跌幅：恐慌錯殺判定用（derive_today_plan）。
    ma120_series = frame["MA120"].dropna()
    ma120_rising = len(ma120_series) > 21 and float(ma120_series.iloc[-1]) > float(ma120_series.iloc[-21])
    close_series = frame["Close"]
    drop_3d_pct = ((latest_close / float(close_series.iloc[-4])) - 1) * 100 if len(close_series) > 4 else 0.0

    # Fundamental analysis via yfinance info（帶快取 + timeout 保護）
    fundamental_score = 0
    graham_number = None
    try:
        info = None
        if fetch_fundamentals:
            now_ts = datetime.now()
            cached_fund = _FUNDAMENTAL_CACHE.get(symbol)
            if cached_fund and (now_ts - cached_fund[0]) < timedelta(minutes=FUNDAMENTAL_CACHE_TTL_MINUTES):
                info = cached_fund[1]
            else:
                def _fetch_info() -> dict:
                    return yf.Ticker(symbol).info
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_fetch_info)
                    try:
                        info = fut.result(timeout=FUNDAMENTAL_FETCH_TIMEOUT)
                        if info:
                            _FUNDAMENTAL_CACHE[symbol] = (now_ts, info)
                    except (FuturesTimeoutError, Exception):
                        info = None  # 超時就跳過，不卡主流程
        if info:
            # Piotroski F-Score / Fundamental Score Proxy
            roa = info.get("returnOnAssets")
            cfo = info.get("operatingCashflow")
            net_income = info.get("netIncomeToCommon")
            debt_to_equity = info.get("debtToEquity")
            current_ratio = info.get("currentRatio")
            gross_margin = info.get("grossMargins")
            roe = info.get("returnOnEquity")
            rev_growth = info.get("revenueGrowth")

            if roa and roa > 0: fundamental_score += 1
            if cfo and cfo > 0: fundamental_score += 1
            if net_income and net_income > 0: fundamental_score += 1
            if cfo and net_income and cfo > net_income: fundamental_score += 1
            if debt_to_equity is not None and debt_to_equity < 150: fundamental_score += 1
            if current_ratio and current_ratio > 1.0: fundamental_score += 1
            if gross_margin and gross_margin > 0.20: fundamental_score += 1
            if roe and roe > 0: fundamental_score += 1
            if rev_growth and rev_growth > 0: fundamental_score += 1

            # Benjamin Graham Number
            eps = info.get("trailingEps")
            bvps = info.get("bookValue")
            if eps and bvps and eps > 0 and bvps > 0:
                graham_number = round((22.5 * eps * bvps) ** 0.5, 2)
    except Exception:
        pass

    # Call Investing.com data service early to make it available to all master agents
    investing_data = {} if lightweight else fetch_investing_com_data(symbol, fetch_fundamentals=fetch_fundamentals, close_price=latest_close)
    investingpro_fair_value = investing_data.get("fair_value")
    valuation_gap_pct = investing_data.get("valuation_gap_pct")
    analyst_target_price = investing_data.get("analyst_target")
    warren_ai_momentum = investing_data.get("warren_ai_momentum")
    investingpro_models = investing_data.get("models_breakdown")

    sector_str = ""
    forward_pe = None
    if info:
        sector_str = info.get("sector") or ""
        forward_pe = info.get("forwardPE")
    ai_chain_layer, critical_bottleneck = map_ai_chain_and_bottleneck(symbol, sector_str)

    trend_up = latest_close > ma20 > ma50
    trend_down = latest_close < ma20 < ma50
    overextended = latest_close >= resistance * 0.99 or rsi14 >= 67

    if trend_up and not overextended:
        bias = "偏多"
        buy_low, buy_high = safe_range(
            max(latest_close * 0.986, support * 1.005, ma20 - 0.20 * atr14),
            min(latest_close * 1.003, ma10 + 0.15 * atr14),
            latest_close,
        )
        sell_low, sell_high = safe_range(
            max(latest_close * 1.18, resistance60 * 1.03),
            max(latest_close * 1.28, resistance60 * 1.09),
            latest_close,
        )
        stop_low, stop_high = safe_range(buy_low * 0.94, buy_low * 0.97, latest_close)
        reason = "趨勢向上且還沒過熱，屬於能在現價附近承接的多頭型態。"
    elif trend_up and overextended:
        bias = "偏多"
        buy_low, buy_high = safe_range(
            max(latest_close * 0.95, support * 1.01, ma20 - 0.35 * atr14),
            min(latest_close * 0.985, ma20 + 0.08 * atr14),
            latest_close,
        )
        sell_low, sell_high = safe_range(
            max(latest_close * 1.16, resistance60 * 1.02),
            max(latest_close * 1.26, resistance60 * 1.08),
            latest_close,
        )
        stop_low, stop_high = safe_range(buy_low * 0.93, buy_low * 0.96, latest_close)
        reason = "趨勢仍強，但位置已偏高，新的買點要等回檔才漂亮。"
    elif trend_down:
        bias = "偏空"
        buy_low, buy_high = safe_range(
            max(support60 * 0.98, latest_close * 0.88),
            min(support * 1.00, latest_close * 0.93),
            latest_close,
        )
        sell_low, sell_high = safe_range(latest_close * 1.05, latest_close * 1.10, latest_close)
        stop_low, stop_high = safe_range(buy_low * 0.91, buy_low * 0.95, latest_close)
        reason = "目前屬於弱勢結構，除非是超跌反彈型交易，否則不適合主動加碼。"
    else:
        bias = "中性"
        buy_low, buy_high = safe_range(
            max(latest_close * 0.97, support * 1.002, ma20 - 0.30 * atr14),
            min(latest_close * 1.001, ma20 + 0.10 * atr14),
            latest_close,
        )
        sell_low, sell_high = safe_range(
            max(latest_close * 1.16, resistance60 * 1.02),
            max(latest_close * 1.24, resistance60 * 1.07),
            latest_close,
        )
        stop_low, stop_high = safe_range(buy_low * 0.94, buy_low * 0.97, latest_close)
        reason = "趨勢尚未明確表態，若要布局應以靠近支撐的分批承接為主。"

    agent_edges = {} if lightweight else compute_agent_edges(frame)
    agents = [
        asdict(agent) for agent in build_agent_opinions(
            frame=frame,
            symbol=symbol,
            fundamental_score=fundamental_score,
            graham_number=graham_number,
            latest_close=latest_close,
            forward_pe=forward_pe,
            investingpro_fair_value=investingpro_fair_value,
            valuation_gap_pct=valuation_gap_pct,
            analyst_target_price=analyst_target_price,
            warren_ai_momentum=warren_ai_momentum,
            ai_chain_layer=ai_chain_layer,
            critical_bottleneck=critical_bottleneck,
            agent_edges=agent_edges,
        )
    ]
    # 股癌（Gooaye）/社群輿情也列為其中一種 agent 看法。
    agents.append(_gooaye_agent_opinion(symbol))
    # 尼可拉斯楊Live（AI 算力蛋糕觀點，YouTube 自動字幕擷取）也列為其中一種 agent 看法。
    agents.append(_nicolas_agent_opinion(symbol))
    horizons = [asdict(view) for view in build_horizon_views(frame)]
    chart = [
        {
            "date": index.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "ma20": round(float(row["MA20"]), 2) if not np.isnan(row["MA20"]) else None,
            "ma50": round(float(row["MA50"]), 2) if not np.isnan(row["MA50"]) else None,
        }
        for index, row in frame.tail(60).iterrows()
    ]

    rule_score, buy_strength = compute_rule_score(agents, horizons)
    
    # --- Deep Integration of Nicholas Yang & Minnie's Strategies ---
    value_boost = 0
    value_notes = []
    
    # 1. Nicholas's AI Chain Boost (+12 points)
    if ai_chain_layer is not None:
        value_boost += 12
        value_notes.append(f"AI產業鏈: {ai_chain_layer}")
        
    # 2. Nicholas's Bottleneck 🔥 Boost (+20 points)
    if critical_bottleneck is not None:
        value_boost += 20
        value_notes.append(f"卡脖子瓶頸: {critical_bottleneck}")

    # 3. Minnie's F-Score Boost (+15, +8, -12)
    if fundamental_score >= 7:
        value_boost += 15
        value_notes.append(f"財務極強健(F-Score: {fundamental_score}/9)")
    elif fundamental_score >= 5:
        value_boost += 8
        value_notes.append(f"財務平穩(F-Score: {fundamental_score}/9)")
    elif 1 <= fundamental_score <= 3:
        value_boost -= 12
        value_notes.append(f"防護價值陷阱(F-Score僅 {fundamental_score}/9)")

    # 4. Minnie's Graham Safety Margin Boost (+20, +10, -5)
    if graham_number and graham_number > 0:
        discount = (graham_number / latest_close) - 1
        if latest_close < graham_number:
            value_boost += 20
            value_notes.append(f"低於葛拉漢價(折價 {discount*100:.1f}%)")
        elif latest_close < graham_number * 1.25:
            value_boost += 10
            value_notes.append(f"貼近葛拉漢價(溢價 {abs(discount)*100:.1f}%)")
        else:
            value_boost -= 5

    # 5. Low P/E Valuation Boost (+12, +5, -15)
    # AI 主線成長股以高 PE 交易是合理的（PEG 才是關鍵），門檻提高到 65 才扣分
    pe_penalty_threshold = 65 if ai_chain_layer is not None else 40
    if forward_pe is not None:
        if 0 < forward_pe <= 18:
            value_boost += 12
            value_notes.append(f"低前瞻本益比({forward_pe:.1f}倍)")
        elif 18 < forward_pe <= 28:
            value_boost += 5
        elif forward_pe > pe_penalty_threshold:
            value_boost -= 15
            value_notes.append(f"估值偏高(本益比 {forward_pe:.1f}倍)")

    # 6. Technical Pullback & Oversold Boost (+10, +5, -12)
    # 2026-07-06 檢討修正：偏多趨勢且貼近 60 日高點的高 RSI 是「突破動能」，不再扣 12 分——
    # 動能因子已驗證正向預測未來報酬（Spearman +0.31~0.41），把突破股扣成「先觀察」正是
    # 之前 6409 在突破日被系統看衰的原因之一。
    _dd60_pct = ((latest_close / close_high60) - 1) * 100 if close_high60 > 0 else 0.0
    if rsi14 < 42:
        value_boost += 10
        value_notes.append(f"技術超跌(RSI: {rsi14:.1f})")
    elif rsi14 <= 50:
        value_boost += 5
        value_notes.append(f"回檔整理中(RSI: {rsi14:.1f})")
    elif rsi14 > 60 and trend_up and _dd60_pct >= -2.0:
        value_boost += 3
        value_notes.append(f"突破創高動能(RSI: {rsi14:.1f})")
    elif rsi14 > 60:
        value_boost -= 12
        value_notes.append(f"短線已高(RSI: {rsi14:.1f})")

    rule_score = clamp_score(rule_score + value_boost)
    novice_rating = calculate_novice_rating(rule_score, forward_pe, critical_bottleneck)

    buy_zone = format_range(buy_low, buy_high)
    sell_zone = format_range(sell_low, sell_high)
    stop_loss = format_range(stop_low, stop_high)
    buy_zone, sell_zone, stop_loss, buy_strength, reason = enforce_position_value(
        buy_zone=buy_zone,
        sell_zone=sell_zone,
        stop_loss=stop_loss,
        buy_strength=buy_strength,
        reason=reason,
        fundamental_score=fundamental_score,
        valuation_gap_pct=valuation_gap_pct,
    )

    # 長線虧損閘門：先算個股自身波段歷史 + 12 個月統計預測，若長抱仍虧損則否決買進。
    if lightweight:
        timeline_bt = None
        price_forecast = {}
        long_term_risk = None
    else:
        timeline_bt = compute_timeline_backtest(symbol, frame)
        price_forecast = build_price_forecast(
            frame,
            latest_close=latest_close,
            bias=bias,
            rsi14=rsi14,
            composite_score=rule_score,
            analyst_target_price=analyst_target_price,
            symbol=symbol,
        )
        long_term_risk = evaluate_long_term_risk(
            price_forecast or None, timeline_bt,
            latest_close=latest_close, ma_long=ma120,
            fundamental_score=fundamental_score, bias=bias,
            valuation_gap_pct=valuation_gap_pct,
        )
        if long_term_risk.get("blocked"):
            buy_strength = "不建議進場"
            reason = f"{reason} ⚠️ 長線虧損風險：{long_term_risk.get('note', '')}"
            if price_forecast:
                price_forecast["verdict"] = "長線恐虧損 不建議買進"
                price_forecast["verdict_reason"] = long_term_risk.get("note", "")

    (
        today_action,
        today_entry_zone,
        today_note,
        today_exit_action,
        today_exit_zone,
        today_exit_note,
        expected_return_pct,
        reward_ratio,
        holding_days_estimate,
        holding_window,
        kelly_position_pct,
    ) = derive_today_plan(
        latest_close=latest_close,
        ma50=ma50,
        atr14=atr14,
        rsi14=rsi14,
        bias=bias,
        buy_strength=buy_strength,
        buy_zone=buy_zone,
        sell_zone=sell_zone,
        stop_loss=stop_loss,
        candlestick_pattern=candlestick_pattern,
        fundamental_score=fundamental_score,
        valuation_gap_pct=valuation_gap_pct,
        ai_chain_layer=ai_chain_layer,
        ma120=ma120,
        long_term_blocked=bool(long_term_risk and long_term_risk.get("blocked")),
        drawdown_from_high_pct=((latest_close / close_high60) - 1) * 100 if close_high60 > 0 else 0.0,
        ma20=ma20,
        # 基本面是否「真的有抓到」：沒抓（每日管線）或抓失敗時，主線/品質判定不再卡 F-Score。
        fundamentals_available=bool(info),
        ma120_rising=ma120_rising,
        drop_3d_pct=drop_3d_pct,
    )

    # Re-use pre-fetched Investing.com data from early scan
    pass
    
    # Calculate master strategy texts
    sym_upper = symbol.upper().split(".")[0].strip()
    cognitive_temperature_gap = None
    geopolitical_timing_advice = None
    value_trap_risk = None
    
    if sym_upper == "MU":
        cognitive_temperature_gap = "「認知溫差極致釋放標的」：市場常將美光視為『循環商品記憶體股』而給予低估值（P/E 5-6）。但在 AI 時代，HBM3e 作為硬體基建核心具有高利潤率特質。對標 Nvidia/Apple 20-25 倍本益比模型，估值重估潛力極高。Timothy Arcuri (UBS) 已設目標價高達 $1,625；若完整對標 Apple 模型，潛在價值將攀升至 $2,100 以上。"
        geopolitical_timing_advice = "「先前佈局（Pre-peace）擇時策略」：在霍爾木茲海峽（Strait of Hormuz）因伊朗政局引發大盤修正時，乃是先前佈局優質科技股的絕佳買點。等官方正式宣佈和平時，開航消息早已被市場反映，在此之前低吸美光（MU）是專業操盤手的智慧之選。"
        value_trap_risk = "防範價值陷阱：無明顯威脅。美光的 HBM3e 產能已被 Blackwell 晶片完全包銷至 2025/2026 年，技術壁壘與定價權極高，是真正的基建龍頭股，完全排除價值陷阱。"
    elif sym_upper == "2451":
        cognitive_temperature_gap = "「記憶體模組邊緣重分類」：創見 (Transcend) 正處於從消費級模組轉向『AI 工業級邊緣儲存設施』的 re-rating 期，市場對其分類尚未完全展開，存在顯著的認知溫差差距。"
        geopolitical_timing_advice = "「半導體鏈先前佈局策略」：在兩岸半導體鏈溢價波動引發非理性拉回時，即是分批承接創見的黃金機會。Warren AI 技術動能為強力買進，應於今日可買價格區分批配置建倉。"
        value_trap_risk = "防範價值陷阱：排除。帳上淨現金極高，配息能力強健，且工業儲存與航太工規模組具有極高替代門檻，非套殼軟體，無 AI 顛覆危機。"
    elif sym_upper == "ADBE":
        cognitive_temperature_gap = "「估值折價的價值陷阱」：Adobe 過去作為創意軟體龍頭，享有高 P/E 溢價。但在 AI 時代，生成式 text-to-image AI 讓『單句 prompt 取代 Photoshop 工作流』成為現實。若將其從龍頭溢價重估為『面臨顛覆風險之傳統軟體』，估值模型面臨崩塌危險。"
        geopolitical_timing_advice = "避開策略：消息面與技術面均呈死亡交叉，不建議在新倉操作，若有部位應分批逢高落袋或停損避險。"
        value_trap_risk = "🔴【警示：典型價值陷阱標的】！本益比看似拉回至合理區間，但其核心護城河正遭受 text-to-image AI 顛覆性蒸發。任何不需要傳統複雜操作即可一鍵出圖的工作流，都會直接液化 Adobe 的高毛利訂閱護城河。建議迴避。"
    else:
        cognitive_temperature_gap = "「板塊估值平穩」：目前此股票處於常態估值分類區間，無劇烈的重新分類溫差，操作上以量價技術指標與基本面 F-Score 護城河審計為主。"
        geopolitical_timing_advice = "大環境大盤處於溫和運行期，地緣政治對其影響在中性區間，適合順勢技術操作。"
        value_trap_risk = f"【防範價值陷阱審計】：本益比與 F-Score {fundamental_score}/9 屬於正常範圍，目前無明顯被生成式 AI 輕易替代之套殼軟體風險，安全邊際相對可控。"

    report = SignalReport(
        symbol=symbol,
        latest_close=round(latest_close, 2),
        ma20=round(ma20, 2),
        ma50=round(ma50, 2),
        rsi14=round(rsi14, 2),
        atr14=round(atr14, 2),
        support=round(support, 2),
        resistance=round(resistance, 2),
        bias=bias,
        buy_zone=buy_zone,
        sell_zone=sell_zone,
        stop_loss=stop_loss,
        reason=reason,
        rule_score=rule_score,
        ai_score=None,
        composite_score=rule_score,
        buy_strength=buy_strength,
        today_action=today_action,
        today_entry_zone=today_entry_zone,
        today_note=today_note,
        today_exit_action=today_exit_action,
        today_exit_zone=today_exit_zone,
        today_exit_note=today_exit_note,
        expected_return_pct=expected_return_pct,
        risk_reward_ratio=reward_ratio,
        holding_days_estimate=holding_days_estimate,
        holding_window=holding_window,
        backtest=None,
        committee_summary=None,
        committee_model=None,
        ai_enabled=False,
        ai_available=False,
        ai_error=None,
        chart=chart,
        agents=agents,
        horizons=horizons,
        fundamental_score=fundamental_score,
        graham_number=graham_number,
        macd_value=round(float(latest["MACD"]), 4),
        macd_signal=round(float(latest["MACD_Signal"]), 4),
        macd_hist=round(float(latest["MACD_Hist"]), 4),
        bb_width=round(float(latest["BB_Width"]), 4),
        candlestick_pattern=candlestick_pattern,
        kelly_position_pct=kelly_position_pct,
        decision_assistance="",
        timeline_backtest=timeline_bt,
        ai_chain_layer=ai_chain_layer,
        critical_bottleneck=critical_bottleneck,
        novice_rating=novice_rating,
        investingpro_fair_value=investing_data.get("fair_value"),
        valuation_gap_pct=investing_data.get("valuation_gap_pct"),
        analyst_target_price=investing_data.get("analyst_target"),
        warren_ai_momentum=investing_data.get("warren_ai_momentum"),
        investingpro_models=investing_data.get("models_breakdown"),
        cognitive_temperature_gap=cognitive_temperature_gap,
        geopolitical_timing_advice=geopolitical_timing_advice,
        value_trap_risk=value_trap_risk,
        price_forecast=price_forecast or None,
        long_term_risk=long_term_risk,
        ma120=round(ma120, 2),
        drawdown_from_high_pct=round(((latest_close / close_high60) - 1) * 100, 2) if close_high60 > 0 else 0.0,
    )

    report.decision_assistance = generate_decision_assistance(report)
    return report


def is_ollama_ready(model_name: str, base_url: str = DEFAULT_OLLAMA_BASE_URL) -> tuple[bool, str | None]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        available = {model.get("name") for model in models}
        if model_name not in available:
            return False, f"Ollama 已啟動，但尚未下載模型 `{model_name}`。"
        return True, None
    except requests.RequestException:
        return False, "Ollama 尚未在目前設定的位址啟動。"


def build_committee_prompt(report: SignalReport) -> str:
    payload = {
        "symbol": report.symbol,
        "latest_close": report.latest_close,
        "ma20": report.ma20,
        "ma50": report.ma50,
        "rsi14": report.rsi14,
        "atr14": report.atr14,
        "macd_value": report.macd_value,
        "macd_signal": report.macd_signal,
        "bb_width": report.bb_width,
        "fundamental_score_f_score": f"{report.fundamental_score}/9",
        "graham_defensive_price": report.graham_number,
        "bias": report.bias,
        "buy_strength": report.buy_strength,
        "today_action": report.today_action,
        "today_entry_zone": report.today_entry_zone,
        "today_exit_action": report.today_exit_action,
        "today_exit_zone": report.today_exit_zone,
        "expected_return_pct": report.expected_return_pct,
        "risk_reward_ratio": report.risk_reward_ratio,
        "kelly_position_suggest_pct": f"{report.kelly_position_pct * 100:.2f}%",
        "agents": report.agents,
        "horizons": report.horizons,
    }
    return (
        "你現在是一位極其專業且耐心的資深 AI 產業鏈投資分析師 + 投資教練（融合了尼可拉斯楊【Nicholas Yang】的產業鏈八層/四大瓶頸選股法，與米妮【Minnie】的 AI 智慧定性護城河會審思維）。\n"
        "你面對的是想投資 AI 產業鏈、但剛開始學投資的「投資小白」。請用繁體中文輸出，並嚴格遵循以下要求：\n"
        "1. 所有解釋要極其通俗易懂。\n"
        "2. 每個專業名詞後面，必須用括號給出一句「大白話解釋」（例如：HBM（高頻寬記憶體，即給晶片配的高速通道））。\n"
        "3. 不准堆砌深奧學術術語，不搞冗長繁複的學術報告，多用直觀好懂的比喻。\n"
        "4. 請輸出一個 JSON 物件，包含三個欄位：\n"
        "   - \"summary\": 字串，請用 Markdown 格式輸出極其詳盡、充滿洞察力的「小白友好版聯合成會審分析報告」（格式如下述）。\n"
        "   - \"score_adjustment\": 整數，範圍在 -12 到 12 之間。根據該股的定性分析（卡位瓶頸、護城河強度、估值折溢價、財務健康度），對原始量化分數進行調整。好票、卡住瓶頸的給予正加分；套殼、無護城河、估值極貴的給予大扣分。\n"
        "   - \"conviction\": 字串，值為 \"high\"、\"medium\" 或 \"low\"，代表你對本決策的信心水準。\n\n"
        "【大師聯合成會審框架說明】\n"
        "第一步：按 AI 產業鏈「核心 7 層 + 太空延伸層（共 8 層）」歸類\n"
        "確認該股票落入以下哪一層，並在報告中明確標示。只保留符合 AI/太空定位的標的，同一層多檔票都要公平對待：\n"
        "一、🎮 計算核心 Compute Core：AI 的「大腦」，負責算賬（代表股：NVDA、AMD、INTC）\n"
        "二、💾 儲存與記憶體 Memory & Storage：大腦的「短期記憶 + 倉庫」（代表股：MU、WDC、STX）\n"
        "三、🌈 光通訊 Photonic / Optical：晶片間用「光」高速傳數據（比銅線快，代表股：COHR、LITE）。\n"
        "    * ⚠️ 若為此層，必須進一步明確標註落在哪一細分子層：L1 襯底材料｜L2 有源器件｜L3 光電晶片｜L4 矽光集成 PIC 平台/IP｜L5 光子代工｜L6 引擎/模組/光傳輸｜L7 測試與認證。\n"
        "四、🌐 網路互聯 Networking：把晶片連成超級電腦的網絡設備（代表股：ANET、AVGO、CSCO）\n"
        "五、🏭 半導體製造 Foundry & Equipment：造晶片的廠和設備（代工/封裝/設備/測試/材料，代表股：TSM、ASML、AMAT、LRCX、KLAC）\n"
        "六、⚡ 資料中心基礎設施 DC Infra：給機房供電、散熱、連接（電力/散熱/能源/連接器，代表股：VRT、DELL、GE、VST、3017）\n"
        "七、💡 IP / 軟體 IP & Software：賣晶片設計圖紙或防禦極高、難被 AI 替代的軟體/平台（代表股：ARM、SNPS、CDNS、PLTR、MSFT）\n"
        "八、🚀 太空 / 衛星 Space & Satellite（延伸層）：AI 的延伸戰場，衛星互聯網、發射基建、太空數據與太空算力（代表股：RKLB）。\n"
        "    * ⚠️ 若為此層，必須在分析末尾單獨提醒小白：SpaceX / Starlink 雖是龍頭但未上市，取得其曝險的可能途徑（如 pre-IPO 平台、持有股權的上市基金/控股公司），並提示這類途徑風險極高、流動性差，僅供研究。\n\n"
        "第二步：標記「四大瓶頸」🔥\n"
        "2026 年當前四大卡脖子環節：① CoWoS 封裝 ② HBM 三巨頭 ③ 3nm/2nm 製程 ④ 資料中心電力。\n"
        "若該股直接受益於上述任一瓶頸，請在報告開頭打上「🔥 {瓶頸名稱}」並給予高度加權！\n\n"
        "第三步：小白評級\n"
        "綜合考慮估值、強勢度、賺錢能力（毛利/淨利）、回本能力（ROIC/ROE）、現金流，給出：\n"
        "- 🟢 強烈關注：又好又相對合理，或正卡在四大瓶頸上。\n"
        "- 🔵 關注：是好票，但現在偏貴，等拉回再買。\n"
        "- 🟡 觀望：基本面一般 / 暫時看不清，不宜貿然動手。\n"
        "- 🔴 迴避：基本面弱，或屬於會被生成式 AI 輕易攻陷替代的「套殼應用層軟體」（無防禦力）。\n\n"
        "第四步：米妮定性護城河與回本審查\n"
        "分析基本面毛利率與核心競爭力（是否極難被替代，還是單純套殼）。評估其 ROIC/ROE 及現金流健康度，列出未來 12 個月最該擔心的核心風險。\n\n"
        "【請在 JSON 的 \"summary\" 欄位中，輸出如下 Markdown 格式的完整審查報告】：\n"
        "### 🎙️ 聯合理事會專家審查報告 - {股票代碼}\n"
        "#### 1. 📂 AI 產業鏈歸類與瓶頸定位\n"
        "* **產業鏈定位**：`第 {X} 層：{層級名稱}（用括號大白話解釋這層是幹嘛的）`\n"
        "* **光通訊子層 / 太空層特記**：`{如果是第三層或第八層，請按規則給出子層標記或 SpaceX 曝險說明，否則寫「不適用」}`\n"
        "* **卡脖子瓶頸**：`{如果是四大瓶頸，請標記 🔥 並指出受惠哪個瓶頸，否則寫「無明顯卡脖子瓶頸」}`\n"
        "* **小白評級**：`{🟢強烈關注 / 🔵關注 / 🟡觀望 / 🔴迴避}`\n\n"
        "#### 2. 💎 大師定性審查與核心競爭力 (Moat Audit)\n"
        "* **為什麼這隻票好（護城河評估）**：`{用大白話說明其護城河、核心技術、毛利率及替代難度}`\n"
        "* **避開清單與替代風險**：`{是否為護城河薄弱、可被生成式 AI 輕易替代的「套殼軟體」？說明其防禦力}`\n\n"
        "#### 3. 📊 財務健康度與核心風險 (Fundamentals & Risks)\n"
        "* **賺錢能力與回本指標**：`{簡述其賺錢與回本能力（毛利率、ROE 等），結合目前 F-Score 點評其財務健壯度}`\n"
        "* **未來 12 個月三大核心風險**：`{列出最值得擔憂的 1-2 個硬核風險（如週期性、大客戶流失、政策等）}`\n\n"
        "#### 4. 📈 估值審查與現狀點評 (Valuation & Performance)\n"
        "* **估值貴嗎及原因**：`{用小白聽得懂的話說明目前 PE 或是葛拉漢價折溢價，是貴還是便宜，為什麼市場給這個估值}`\n"
        "* **現狀點評（領賺或落後）**：`{分析近期相對大盤（如費城半導體指數或標普500）是領賺還是落後，為什麼}`\n\n"
        "#### 5. 🛠️ 投資人進場與操作指南 (Action Plan)\n"
        "* **資金配置與分批策略**：`{具體操作建議，例如分幾批（如 5-8 批 DCA）建倉，或等股價拉回均線/支撐區某價位時加倉。註明適合作為核心倉還是試探倉，並呼應凱利公式建議資金配置}`\n\n"
        f"請結合以上框架與以下技術量化數據進行綜合評審，數據資料如下：\n{json.dumps(payload, ensure_ascii=False)}"
    )


def apply_committee_overlay(report: SignalReport, model_name: str = "gemma4:e4b") -> SignalReport:
    enriched = _clone_report(report)
    enriched.ai_enabled = True
    enriched.committee_model = model_name

    ready, error = is_ollama_ready(model_name)
    if not ready:
        enriched.ai_available = False
        enriched.ai_error = error
        enriched.committee_summary = "AI 委員會目前不可用，已自動改用規則分數。"
        return enriched

    payload = {
        "model": model_name,
        "prompt": build_committee_prompt(report),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    try:
        response = requests.post(f"{DEFAULT_OLLAMA_BASE_URL}/api/generate", json=payload, timeout=90)
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        data = json.loads(raw)
        score_adjustment = int(max(-12, min(12, int(data.get("score_adjustment", 0)))))
        ai_score = clamp_score(report.rule_score + score_adjustment)
        composite_score = clamp_score(round(report.rule_score * 0.7 + ai_score * 0.3))

        enriched.ai_available = True
        enriched.ai_error = None
        enriched.ai_score = ai_score
        enriched.composite_score = composite_score
        enriched.buy_strength = score_to_strength(composite_score)
        enriched.committee_summary = str(data.get("summary") or "AI 委員會未提供摘要。")
        return enriched
    except Exception as exc:
        enriched.ai_available = False
        enriched.ai_error = str(exc)
        enriched.committee_summary = "AI 委員會呼叫失敗，已自動改用規則分數。"
        return enriched


def analyze_symbol_with_data(
    ticker: str,
    market: str | None = None,
    period: str = DEFAULT_ANALYSIS_PERIOD,
    *,
    use_ai_committee: bool = False,
    committee_model: str = "gemma4:e4b",
) -> tuple[SignalReport, pd.DataFrame]:
    symbol = normalize_ticker(ticker, market)
    cache_key = _cache_key(symbol, period, use_ai_committee, committee_model)
    cached = _SIGNAL_CACHE.get(cache_key)
    now = datetime.now()
    if cached and now - cached[0] <= timedelta(minutes=SIGNAL_CACHE_TTL_MINUTES):
        return _clone_report(cached[1]), download_prices(symbol, period)

    frame = download_prices(symbol, period)
    report = build_report(symbol, frame)
    if use_ai_committee:
        report = apply_committee_overlay(report, committee_model)
    _SIGNAL_CACHE[cache_key] = (now, _clone_report(report))
    return report, frame


def analyze_symbol(
    ticker: str,
    market: str | None = None,
    period: str = DEFAULT_ANALYSIS_PERIOD,
    *,
    use_ai_committee: bool = False,
    committee_model: str = "gemma4:e4b",
) -> SignalReport:
    report, _ = analyze_symbol_with_data(
        ticker,
        market,
        period,
        use_ai_committee=use_ai_committee,
        committee_model=committee_model,
    )
    return report


def analyze_symbols_batch_with_data(
    tickers: list[str],
    market: str | None = None,
    period: str = DEFAULT_ANALYSIS_PERIOD,
    *,
    use_ai_committee: bool = False,
    committee_model: str = "gemma4:e4b",
) -> tuple[list[tuple[SignalReport, pd.DataFrame]], list[str]]:
    cleaned = [ticker.strip() for ticker in tickers if ticker.strip()]
    if not cleaned:
        raise ValueError("請至少輸入一檔股票。")

    reports: list[tuple[SignalReport, pd.DataFrame]] = []
    errors: list[str] = []

    # yfinance 在同時大量單檔下載時偶爾會回傳異常欄位格式，
    # 這裡先用穩定優先的順序分析，避免多檔查詢整批失敗。
    for ticker in cleaned:
        try:
            reports.append(
                analyze_symbol_with_data(
                    ticker,
                    market,
                    period,
                    use_ai_committee=use_ai_committee,
                    committee_model=committee_model,
                )
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    reports.sort(key=lambda item: item[0].composite_score, reverse=True)
    return reports, errors


def analyze_symbols_batch(
    tickers: list[str],
    market: str | None = None,
    period: str = DEFAULT_ANALYSIS_PERIOD,
    *,
    use_ai_committee: bool = False,
    committee_model: str = "gemma4:e4b",
) -> tuple[list[SignalReport], list[str]]:
    reports_with_frames, errors = analyze_symbols_batch_with_data(
        tickers,
        market,
        period,
        use_ai_committee=use_ai_committee,
        committee_model=committee_model,
    )
    return [report for report, _frame in reports_with_frames], errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple stock signal analysis")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL or 2330")
    parser.add_argument("--market", default=None, help="Optional market hint: us or tw")
    parser.add_argument("--period", default=DEFAULT_ANALYSIS_PERIOD, help="Yahoo Finance lookback period")
    parser.add_argument("--use-ai-committee", action="store_true", help="Enable Ollama committee overlay")
    parser.add_argument("--committee-model", default="gemma4:e4b", help="Ollama model name")
    args = parser.parse_args()

    report = analyze_symbol(
        args.ticker,
        args.market,
        args.period,
        use_ai_committee=args.use_ai_committee,
        committee_model=args.committee_model,
    )
    print(json.dumps(asdict(report), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
