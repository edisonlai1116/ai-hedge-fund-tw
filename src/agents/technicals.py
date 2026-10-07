import math

import json
import pandas as pd
import numpy as np



def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling NaN cases
    
    Args:
        value: The value to convert (can be pandas scalar, numpy value, etc.)
        default: Default value to return if the input is NaN or invalid
    
    Returns:
        float: The converted value or default if NaN/invalid
    """
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


##### Technical Analyst #####
def technical_analyst_agent(state, agent_id: str = "technical_analyst_agent"):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    # 依賴 langchain / 付費資料 API 的匯入放在這裡：讓下方純價格計算函式可在免 Key 環境（策略排名、回測）單獨使用。
    from langchain_core.messages import HumanMessage

    from src.graph.state import show_agent_reasoning
    from src.tools.api import get_prices, prices_to_df
    from src.utils.api_key import get_api_key_from_state
    from src.utils.progress import progress

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        progress.update_status(agent_id, ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status(agent_id, ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status(agent_id, ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        progress.update_status(agent_id, ticker, "Combining signals")
        components = {
            "trend": trend_signals,
            "mean_reversion": mean_reversion_signals,
            "momentum": momentum_signals,
            "volatility": volatility_signals,
            "stat_arb": stat_arb_signals,
        }
        combined_signal = weighted_signal_combination(components, TECHNICAL_WEIGHTS)

        def _view(sig):
            return {
                "signal": sig["signal"],
                "confidence": round(sig["confidence"] * 100),
                "score": sig.get("score"),
                "metrics": normalize_pandas(sig["metrics"]),
            }

        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "technical_score": combined_signal["score"],
            "bars_available": len(prices_df),
            "reasoning": {
                "trend_score": _view(trend_signals),
                "momentum_score": _view(momentum_signals),
                "mean_reversion_score": _view(mean_reversion_signals),
                "volatility_regime_score": {**_view(volatility_signals), "note": "無方向，僅供部位大小/風控"},
                "statistical_experimental": {**_view(stat_arb_signals), "note": "experimental，不計入總分"},
                "components_used": combined_signal["components_used"],
            },
        }
        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(technical_analysis, indent=4))

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = technical_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


INSUFFICIENT = "insufficient_data"

# 各子訊號所需的最少 K 線數（不足就回 insufficient_data，不得偷偷給方向、也不當 neutral 參與加權）。
MIN_BARS = {
    "trend": 110,           # EMA55 需約 2 倍週期暖機；ADX/MACD 也需要
    "mean_reversion": 60,   # MA50 / 50 日標準差 + 緩衝
    "momentum": 127,        # 126 日複利報酬
    "volatility": 85,       # 21 日波動 + 63 日均值
    "stat_arb": 64,         # 63 日偏態/峰態
}

# 正式總分只用有方向意義的子模組；volatility 只描述環境（無方向）、stat_arb(Hurst) 標記 experimental。
TECHNICAL_WEIGHTS = {"trend": 0.40, "momentum": 0.40, "mean_reversion": 0.20}


def _insufficient(needed: int, have: int) -> dict:
    return {
        "signal": INSUFFICIENT,
        "confidence": 0.0,
        "score": None,
        "metrics": {"bars_required": needed, "bars_available": have},
    }


def _score_to_signal(score: float) -> tuple[str, float]:
    """0~100 分 → 方向與信心（50 為中性；距離 50 越遠信心越高）。"""
    if score >= 60:
        return "bullish", min(1.0, (score - 50) / 50)
    if score <= 40:
        return "bearish", min(1.0, (50 - score) / 50)
    return "neutral", 0.5


def compounded_return(close: pd.Series, n: int) -> float | None:
    """(1+r1)(1+r2)...(1+rn)-1，等同 close[-1]/close[-n-1]-1。資料不足回 None。"""
    c = close.dropna()
    if len(c) < n + 1:
        return None
    return float(c.iloc[-1] / c.iloc[-n - 1] - 1.0)


def calculate_trend_signals(prices_df):
    """趨勢：EMA8/21/55 排列 + MACD + ADX 強度 → trend_score 0~100。"""
    n = len(prices_df)
    if n < MIN_BARS["trend"]:
        return _insufficient(MIN_BARS["trend"], n)
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    adx = calculate_adx(prices_df.copy(), 14)
    macd_line, signal_line, _ = calculate_macd(prices_df)

    close = float(prices_df["close"].iloc[-1])
    points = 0.0
    points += 1.0 if ema_8.iloc[-1] > ema_21.iloc[-1] else -1.0
    points += 1.0 if ema_21.iloc[-1] > ema_55.iloc[-1] else -1.0
    points += 1.0 if close > ema_55.iloc[-1] else -1.0
    points += 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
    adx_val = safe_float(adx["adx"].iloc[-1], 0.0)
    strength = min(adx_val / 40.0, 1.0)            # ADX 40 以上視為強趨勢
    # 方向（-4~+4）× 強度（0.5~1.0）→ 0~100
    score = 50 + (points / 4.0) * 50 * (0.5 + 0.5 * strength)
    signal, confidence = _score_to_signal(score)
    return {
        "signal": signal,
        "confidence": confidence,
        "score": round(score, 1),
        "metrics": {
            "adx": adx_val,
            "ema_alignment_points": points,
            "close_vs_ema55_pct": safe_float((close / ema_55.iloc[-1] - 1) * 100),
            "macd_bullish": bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """均值回歸：只在極端（z-score 超過 ±2 且在布林帶外緣）時給方向，其餘 50。"""
    n = len(prices_df)
    if n < MIN_BARS["mean_reversion"]:
        return _insufficient(MIN_BARS["mean_reversion"], n)
    close = prices_df["close"]
    ma_50 = close.rolling(50).mean()
    std_50 = close.rolling(50).std()
    z = safe_float(((close - ma_50) / std_50).iloc[-1])
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
    width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    price_vs_bb = safe_float((close.iloc[-1] - bb_lower.iloc[-1]) / width, 0.5) if width else 0.5
    rsi_14 = safe_float(calculate_rsi(prices_df, 14).iloc[-1], 50.0)

    score = 50.0
    if z < -2 and price_vs_bb < 0.2:
        score = 50 + min(abs(z) / 4, 1.0) * 40
    elif z > 2 and price_vs_bb > 0.8:
        score = 50 - min(abs(z) / 4, 1.0) * 40
    signal, confidence = _score_to_signal(score)
    return {
        "signal": signal,
        "confidence": confidence,
        "score": round(score, 1),
        "metrics": {"z_score": z, "price_vs_bb": price_vs_bb, "rsi_14": rsi_14},
    }


def calculate_momentum_signals(prices_df):
    """動能：1/3/6 個月「複利」報酬（不是日報酬加總）+ 量能確認 → momentum_score 0~100。"""
    n = len(prices_df)
    if n < MIN_BARS["momentum"]:
        return _insufficient(MIN_BARS["momentum"], n)
    close = prices_df["close"]
    mom_1m = compounded_return(close, 21)
    mom_3m = compounded_return(close, 63)
    mom_6m = compounded_return(close, 126)
    blended = 0.35 * mom_1m + 0.25 * mom_3m + 0.40 * mom_6m
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = safe_float((prices_df["volume"] / volume_ma).iloc[-1], 1.0)

    # ±40% 的混合報酬對應 0/100；量能放大時把分數往方向推 10%。
    score = 50 + max(-1.0, min(1.0, blended / 0.40)) * 50
    if volume_momentum > 1.2:
        score = 50 + (score - 50) * 1.1
    score = max(0.0, min(100.0, score))
    signal, confidence = _score_to_signal(score)
    return {
        "signal": signal,
        "confidence": confidence,
        "score": round(score, 1),
        "metrics": {
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "momentum_6m": mom_6m,
            "volume_momentum": volume_momentum,
            "method": "compounded",
        },
    }


def calculate_volatility_signals(prices_df):
    """波動環境（無方向）：只描述現在是低/正常/高波動，供部位大小與風控使用，
    signal 永遠是 neutral，不參與 bullish/bearish 加權。高波動 ≠ 看空。"""
    n = len(prices_df)
    if n < MIN_BARS["volatility"]:
        return _insufficient(MIN_BARS["volatility"], n)
    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = safe_float((hist_vol / vol_ma).iloc[-1], 1.0)
    atr_ratio = safe_float((calculate_atr(prices_df) / prices_df["close"]).iloc[-1])
    regime = "low" if vol_regime < 0.8 else ("high" if vol_regime > 1.2 else "normal")
    # volatility_regime_score：100 = 非常平靜，0 = 非常動盪（給風控/部位用，不是多空）
    regime_score = max(0.0, min(100.0, 100 - (vol_regime - 0.5) * 66.7))
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "score": round(regime_score, 1),
        "directional": False,
        "metrics": {
            "historical_volatility": safe_float(hist_vol.iloc[-1]),
            "volatility_regime": vol_regime,
            "regime": regime,
            "atr_ratio": atr_ratio,
        },
    }


def calculate_stat_arb_signals(prices_df):
    """統計特徵（EXPERIMENTAL）：Hurst 估計未經嚴謹驗證，只報告數值、不影響正式總分。"""
    n = len(prices_df)
    if n < MIN_BARS["stat_arb"]:
        return _insufficient(MIN_BARS["stat_arb"], n)
    returns = prices_df["close"].pct_change()
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "score": None,
        "experimental": True,
        "metrics": {
            "hurst_exponent_experimental": safe_float(calculate_hurst_exponent(prices_df["close"])),
            "skewness": safe_float(returns.rolling(63).skew().iloc[-1]),
            "kurtosis": safe_float(returns.rolling(63).kurt().iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    """只合併「有分數」的方向性子模組；insufficient_data 直接排除並重新正規化權重，
    不會被當成 neutral 稀釋。全部不足 → insufficient_data。"""
    usable = {k: v for k, v in signals.items() if k in weights and v.get("score") is not None
              and v.get("signal") != INSUFFICIENT}
    if not usable:
        return {"signal": INSUFFICIENT, "confidence": 0.0, "score": None, "components_used": []}
    total_w = sum(weights[k] for k in usable)
    score = sum(usable[k]["score"] * weights[k] for k in usable) / total_w
    signal, confidence = _score_to_signal(score)
    return {"signal": signal, "confidence": confidence, "score": round(score, 1), "components_used": sorted(usable)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5


def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = calculate_ema(prices_df, 12)
    ema_26 = calculate_ema(prices_df, 26)
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def detect_reversal_patterns(prices_df: pd.DataFrame) -> str:
    if len(prices_df) < 2:
        return "neutral"
    latest = prices_df.iloc[-1]
    prev = prices_df.iloc[-2]
    
    o1, h1, l1, c1 = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
    o2, h2, l2, c2 = float(latest["open"]), float(latest["high"]), float(latest["low"]), float(latest["close"])
    
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    
    # 1. Bullish Reversal patterns
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1 and body2 > body1:
        return "bullish"  # Bullish Engulfing
    
    total_range = h2 - l2
    if total_range > 0:
        body_top = max(o2, c2)
        body_bottom = min(o2, c2)
        lower_shadow = body_bottom - l2
        upper_shadow = h2 - body_top
        if lower_shadow > 2 * body2 and upper_shadow < 0.2 * total_range:
            return "bullish"  # Hammer

    # 2. Bearish Reversal patterns
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1 and body2 > body1:
        return "bearish"  # Bearish Engulfing
        
    if total_range > 0:
        body_top = max(o2, c2)
        body_bottom = min(o2, c2)
        lower_shadow = body_bottom - l2
        upper_shadow = h2 - body_top
        if upper_shadow > 2 * body2 and lower_shadow < 0.2 * total_range:
            return "bearish"  # Shooting Star

    return "neutral"
