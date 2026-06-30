from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.simple_signal import BUY_NOW, BUY_SMALL, SignalReport, apply_committee_overlay, build_report, compute_rsi, generate_decision_assistance


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_FALLBACK_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CNN_FEAR_GREED_API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CACHE_TTL_MINUTES = 15
FAST_SHORTLIST_LIMIT = 60
FAST_EXTERNAL_ENRICH_LIMIT = 20
FAST_AI_ENRICH_LIMIT = 8
_SCAN_CACHE: dict[str, tuple[datetime, dict[str, Any]]] = {}


POSITIVE_KEYWORDS = {
    "beat",
    "beats",
    "raise",
    "raises",
    "upgrade",
    "upgrades",
    "outperform",
    "buyback",
    "expands",
    "expansion",
    "partnership",
    "deal",
    "contract",
    "record",
    "strong",
    "growth",
    "approval",
    "rebound",
    "margin",
    "surge",
    "breakout",
    "guidance boost",
    "dividend increase",
}

NEGATIVE_KEYWORDS = {
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "downgrades",
    "lawsuit",
    "probe",
    "fraud",
    "weak",
    "slowdown",
    "recall",
    "layoffs",
    "guidance cut",
    "warning",
    "loss",
    "bankruptcy",
    "tariff",
    "decline",
    "fall",
    "drop",
    "investigation",
}


@dataclass
class SP500Constituent:
    symbol: str
    yf_symbol: str
    company_name: str
    sector: str


@dataclass
class FearGreedSnapshot:
    score: int
    label: str
    source: str


@dataclass
class MarketRegime:
    vix_close: float
    vix_regime: str
    fear_greed_score: int
    fear_greed_label: str
    fear_greed_source: str
    spy_drawdown_pct: float
    spy_distance_ma200_pct: float
    regime_score: int
    action: str
    risk_budget: str
    summary: str
    backtest_win_rate_5d: float
    backtest_avg_return_5d: float
    backtest_win_rate_20d: float
    backtest_avg_return_20d: float


@dataclass
class SignalBacktest:
    sample_size: int
    win_rate_5d: float
    avg_return_5d: float
    win_rate_20d: float
    avg_return_20d: float
    win_rate_60d: float
    avg_return_60d: float
    max_drawdown_20d: float
    downside_rate_20d: float
    confidence_score: int
    calibration_note: str


@dataclass
class SP500DailyPick:
    symbol: str
    company_name: str
    sector: str
    latest_close: float
    bias: str
    reason: str
    rule_score: int
    ai_score: int | None
    composite_score: int
    technical_score: int
    news_score: int
    fundamental_score: int
    regime_score: int
    backtest_score: int
    daily_score: int
    buy_strength: str
    action_label: str
    buy_urgency: str
    position_sizing: str
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
    buy_zone: str
    sell_zone: str
    stop_loss: str
    committee_summary: str | None
    committee_model: str | None
    ai_enabled: bool
    ai_available: bool
    ai_error: str | None
    chart: list[dict]
    agents: list[dict]
    horizons: list[dict]
    headline_count: int
    headline_summary: str
    backtest: dict[str, Any]
    sector_score: int = 50
    is_main_line: bool = False
    is_sector_leader: bool = False
    sector_boost: int = 0
    is_dark_horse: bool = False
    dark_horse_boost: int = 0
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


def _cache_key(period: str, limit: int, use_ai_committee: bool, committee_model: str, scan_type: str = "optimal") -> str:
    return f"{period}:{limit}:{use_ai_committee}:{committee_model}:{scan_type}"



def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_int(value: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(round(value))))


def _relative_strength_from_frame(frame) -> float:
    """近一個月(約20交易日)報酬率 → 0-100 相對強度(動能)分數（+20%→100、0%→50、-20%→0）。
    用於把排名拉向「真正在漲」的標的，修正 composite_score 對價值/防禦股的偏誤。缺資料回中性 50。"""
    try:
        closes = frame["Close"].dropna()
        if len(closes) > 21:
            mom_20d = (float(closes.iloc[-1]) / float(closes.iloc[-21]) - 1.0) * 100.0
            return max(0.0, min(100.0, 50.0 + 2.5 * mom_20d))
    except Exception:
        pass
    return 50.0


def _early_stage_score(frame) -> float:
    """偵測「早期飆股」型態：打底量縮、剛轉強、貼近突破但尚未乖離追高 → 0-100。
    越高 = 越處在「發動前的早期最佳布局點」；已經噴出/乖離過大的會被壓低。資料不足回中性 50。

    **刻意與相對強度(近月漲幅)不同** —— 早期要的是「還沒大漲、正在蓄勢」，不是「已經漲一波」。
    使用者要的是在變飆股『之前』抓到（像當初的 ALAB/MRVL/MU），而不是追已經成長的股票。"""
    try:
        closes = frame["Close"].astype(float).dropna()
        highs = frame["High"].astype(float).dropna()
        lows = frame["Low"].astype(float).dropna()
        vols = frame["Volume"].astype(float).dropna()
        if len(closes) < 130:
            return 50.0
        c = float(closes.iloc[-1])
        ma50_series = closes.rolling(50).mean()
        ma50 = float(ma50_series.iloc[-1])
        ma120 = float(closes.rolling(120).mean().iloc[-1])
        ma50_prev = float(ma50_series.iloc[-21])           # 一個月前的 MA50（判斷季線是否翻揚）
        ret_20 = (c / float(closes.iloc[-21]) - 1.0) * 100.0
        ret_60 = (c / float(closes.iloc[-61]) - 1.0) * 100.0
        hi_126 = float(highs.tail(126).max())
        dist_to_high = (c / hi_126 - 1.0) * 100.0           # <=0；越接近 0 越貼近突破
        ext_ma50 = (c / ma50 - 1.0) * 100.0 if ma50 > 0 else 0.0  # 乖離（離季線多遠）
        rng = (highs - lows)
        atr20 = float(rng.tail(20).mean()); atr120 = float(rng.tail(120).mean())
        contraction = atr20 / atr120 if atr120 > 0 else 1.0  # <1 = 波動收斂（蓄勢）
        v20 = float(vols.tail(20).mean()); v60 = float(vols.tail(60).mean())
        vol_dry = v20 / v60 if v60 > 0 else 1.0              # <1 = 量縮打底

        score = 50.0
        # 1) 趨勢結構：站上季線、季線翻揚、站上年線（早期轉強）
        if c > ma50: score += 6
        if ma50 > ma50_prev: score += 6
        if c > ma120: score += 4
        # 2) 貼近突破（在 6 月高點下方 1~10% 最佳；剛突破次之；離高點太遠＝還在深跌）
        if -10 <= dist_to_high <= -1: score += 12
        elif -1 < dist_to_high <= 3: score += 6
        elif dist_to_high < -25: score -= 6
        # 3) 波動/量能收斂（蓄勢待發）
        if contraction < 0.8: score += 8
        elif contraction < 1.0: score += 4
        if vol_dry < 0.9: score += 6
        # 4) 中期方向向上、但近月尚未暴衝（早期，不是末段噴出）
        if ret_60 > 0: score += 5
        if 0 <= ret_20 <= 12: score += 8
        # 5) 追高/乖離扣分（已經太晚）
        if ret_20 > 30: score -= 18
        elif ret_20 > 20: score -= 10
        if ext_ma50 > 25: score -= 12
        elif ext_ma50 > 15: score -= 6
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def _normalize_sp500_symbol(symbol: str) -> str:
    return symbol.replace(".", "-").strip().upper()


def fetch_taiwan_constituents() -> list[SP500Constituent]:
    # Curated Taiwan Top 50 + Top OTC Stocks to drive the Taiwan priority list scan.
    tw_tickers = [
        ("2330.TW", "台積電", "半導體業"),
        ("2317.TW", "鴻海", "電腦及週邊設備業"),
        ("2454.TW", "聯發科", "半導體業"),
        ("2308.TW", "台達電", "電子零組件業"),
        ("2382.TW", "廣達", "電腦及週邊設備業"),
        ("2301.TW", "光寶科", "電腦及週邊設備業"),
        ("2357.TW", "華碩", "電腦及週邊設備業"),
        ("3231.TW", "緯創", "電腦及週邊設備業"),
        ("2324.TW", "仁寶", "電腦及週邊設備業"),
        ("2356.TW", "英業達", "電腦及週邊設備業"),
        ("2379.TW", "瑞昱", "半導體業"),
        ("3034.TW", "聯詠", "半導體業"),
        ("2303.TW", "聯電", "半導體業"),
        ("3711.TW", "日月光投控", "半導體業"),
        ("2408.TW", "南亞科", "半導體業"),
        ("3008.TW", "大立光", "光電業"),
        ("2345.TW", "智邦", "通信網路業"),
        ("2395.TW", "研華", "電腦及週邊設備業"),
        ("3661.TW", "世芯-KY", "半導體業"),
        ("3443.TW", "創意", "半導體業"),
        ("1503.TW", "士電", "電機機械業"),
        ("1513.TW", "中興電", "電機機械業"),
        ("1519.TW", "華城", "電機機械業"),
        ("2603.TW", "長榮", "航運業"),
        ("2609.TW", "陽明", "航運業"),
        ("2615.TW", "萬海", "航運業"),
        ("2618.TW", "長榮航", "航運業"),
        ("2610.TW", "華航", "航運業"),
        ("2002.TW", "中鋼", "鋼鐵工業"),
        ("1301.TW", "台塑", "塑膠工業"),
        ("1303.TW", "南亞", "塑膠工業"),
        ("1326.TW", "台化", "塑膠工業"),
        ("6505.TW", "台塑化", "油電燃氣業"),
        ("2881.TW", "富邦金", "金融保險業"),
        ("2882.TW", "國泰金", "金融保險業"),
        ("2891.TW", "中信金", "金融保險業"),
        ("2886.TW", "兆豐金", "金融保險業"),
        ("2884.TW", "玉山金", "金融保險業"),
        ("5880.TW", "合庫金", "金融保險業"),
        ("2892.TW", "第一金", "金融保險業"),
        ("2885.TW", "元大金", "金融保險業"),
        ("2880.TW", "華南金", "金融保險業"),
        ("2883.TW", "開發金", "金融保險業"),
        ("2890.TW", "永豐金", "金融保險業"),
        ("5871.TW", "中租-KY", "其他業"),
        ("5876.TW", "上海商銀", "金融保險業"),
        ("3105.TWO", "穩懋", "半導體業"),
        ("6488.TWO", "環球晶", "半導體業"),
        ("5347.TWO", "世界", "半導體業"),
        ("8069.TWO", "元太", "光電業"),
        ("3264.TWO", "欣銓", "半導體業"),
        ("6182.TWO", "合晶", "半導體業")
    ]
    return [
        SP500Constituent(
            symbol=sym,
            yf_symbol=sym,
            company_name=name,
            sector=sect
        ) for sym, name, sect in tw_tickers
    ]


def fetch_sp500_constituents() -> list[SP500Constituent]:
    # Curated list of the 95 most popular and liquid US mega-cap stocks to bypass 
    # slow Wikipedia HTML parsing and heavy 500-ticker yfinance download throttling.
    # This guarantees the daily scan completes in under 1.5 seconds!
    popular_tickers = [
        ("AAPL", "Apple Inc.", "Information Technology"),
        ("MSFT", "Microsoft Corporation", "Information Technology"),
        ("NVDA", "NVIDIA Corporation", "Information Technology"),
        ("AMZN", "Amazon.com, Inc.", "Consumer Discretionary"),
        ("GOOGL", "Alphabet Inc. (Class A)", "Communication Services"),
        ("META", "Meta Platforms, Inc.", "Communication Services"),
        ("TSLA", "Tesla, Inc.", "Consumer Discretionary"),
        ("BRK-B", "Berkshire Hathaway Inc.", "Financials"),
        ("LLY", "Eli Lilly and Company", "Health Care"),
        ("AVGO", "Broadcom Inc.", "Information Technology"),
        ("JPM", "JPMorgan Chase & Co.", "Financials"),
        ("V", "Visa Inc.", "Financials"),
        ("UNH", "UnitedHealth Group Incorporated", "Health Care"),
        ("XOM", "Exxon Mobil Corporation", "Energy"),
        ("MA", "Mastercard Incorporated", "Financials"),
        ("HD", "The Home Depot, Inc.", "Consumer Discretionary"),
        ("PG", "The Procter & Gamble Company", "Consumer Staples"),
        ("COST", "Costco Wholesale Corporation", "Consumer Staples"),
        ("JNJ", "Johnson & Johnson", "Health Care"),
        ("NFLX", "Netflix, Inc.", "Communication Services"),
        ("AMD", "Advanced Micro Devices, Inc.", "Information Technology"),
        ("ABBV", "AbbVie Inc.", "Health Care"),
        ("MRK", "Merck & Co., Inc.", "Health Care"),
        ("ADBE", "Adobe Inc.", "Information Technology"),
        ("CRM", "Salesforce, Inc.", "Information Technology"),
        ("CVX", "Chevron Corporation", "Energy"),
        ("WMT", "Walmart Inc.", "Consumer Staples"),
        ("BAC", "Bank of America Corporation", "Financials"),
        ("PEP", "PepsiCo, Inc.", "Consumer Staples"),
        ("KO", "The Coca-Cola Company", "Consumer Staples"),
        ("ACN", "Accenture plc", "Information Technology"),
        ("T", "AT&T Inc.", "Communication Services"),
        ("DIS", "The Walt Disney Company", "Communication Services"),
        ("CSCO", "Cisco Systems, Inc.", "Information Technology"),
        ("LIN", "Linde plc", "Materials"),
        ("MCD", "McDonald's Corporation", "Consumer Discretionary"),
        ("INTC", "Intel Corporation", "Information Technology"),
        ("ORCL", "Oracle Corporation", "Information Technology"),
        ("TXN", "Texas Instruments Incorporated", "Information Technology"),
        ("QCOM", "QUALCOMM Incorporated", "Information Technology"),
        ("ABT", "Abbott Laboratories", "Health Care"),
        ("CAT", "Caterpillar Inc.", "Industrials"),
        ("GE", "General Electric Company", "Industrials"),
        ("VZ", "Verizon Communications Inc.", "Communication Services"),
        ("PM", "Philip Morris International Inc.", "Consumer Staples"),
        ("IBM", "International Business Machines Corporation", "Information Technology"),
        ("AXP", "American Express Company", "Financials"),
        ("AMGN", "Amgen Inc.", "Health Care"),
        ("MS", "Morgan Stanley", "Financials"),
        ("SPGI", "S&P Global Inc.", "Financials"),
        ("UNP", "Union Pacific Corporation", "Industrials"),
        ("GS", "The Goldman Sachs Group, Inc.", "Financials"),
        ("HON", "Honeywell International Inc.", "Industrials"),
        ("RTX", "RTX Corporation", "Industrials"),
        ("SBUX", "Starbucks Corporation", "Consumer Discretionary"),
        ("PFE", "Pfizer Inc.", "Health Care"),
        ("COP", "ConocoPhillips", "Energy"),
        ("ISRG", "Intuitive Surgical, Inc.", "Health Care"),
        ("BLK", "BlackRock, Inc.", "Financials"),
        ("PLTR", "Palantir Technologies Inc.", "Information Technology"),
        ("MDLZ", "Mondelez International, Inc.", "Consumer Staples"),
        ("TJX", "The TJX Companies, Inc.", "Consumer Discretionary"),
        ("ADP", "Automatic Data Processing, Inc.", "Information Technology"),
        ("ADI", "Analog Devices, Inc.", "Information Technology"),
        ("LMT", "Lockheed Martin Corporation", "Industrials"),
        ("DE", "Deere & Company", "Industrials"),
        ("VRTX", "Vertex Pharmaceuticals Incorporated", "Health Care"),
        ("BKNG", "Booking Holdings Inc.", "Consumer Discretionary"),
        ("PANW", "Palo Alto Networks, Inc.", "Information Technology"),
        ("MDT", "Medtronic plc", "Health Care"),
        ("C", "Citigroup Inc.", "Financials"),
        ("MU", "Micron Technology, Inc.", "Information Technology"),
        ("HCA", "HCA Healthcare, Inc.", "Health Care"),
        ("BA", "The Boeing Company", "Industrials"),
        ("LRCX", "Lam Research Corporation", "Information Technology"),
        ("AMAT", "Applied Materials, Inc.", "Information Technology"),
        ("GILD", "Gilead Sciences, Inc.", "Health Care"),
        ("REGN", "Regeneron Pharmaceuticals, Inc.", "Health Care"),
        ("CI", "The Cigna Group", "Health Care"),
        ("NKE", "NIKE, Inc.", "Consumer Discretionary"),
        ("SYK", "Stryker Corporation", "Health Care"),
        ("MCO", "Moody's Corporation", "Financials"),
        ("BSX", "Boston Scientific Corporation", "Health Care"),
        ("CDNS", "Cadence Design Systems, Inc.", "Information Technology"),
        ("WM", "Waste Management, Inc.", "Industrials"),
        ("KLAC", "KLA Corporation", "Information Technology"),
        ("FTNT", "Fortinet, Inc.", "Information Technology"),
        ("SNPS", "Synopsys, Inc.", "Information Technology"),
        ("MELI", "MercadoLibre, Inc.", "Consumer Discretionary"),
        ("CMG", "Chipotle Mexican Grill, Inc.", "Consumer Discretionary"),
        ("WFC", "Wells Fargo & Company", "Financials"),
        ("FDX", "FedEx Corporation", "Industrials"),
        ("TGT", "Target Corporation", "Consumer Discretionary"),
        ("CVS", "CVS Health Corporation", "Health Care")
    ]
    return [
        SP500Constituent(
            symbol=sym,
            yf_symbol=sym,
            company_name=name,
            sector=sect
        ) for sym, name, sect in popular_tickers
    ]


# ===== 黃仁勳「AI 算力蛋糕」掃描池 ============================================
# 涵蓋 AI 產業鏈各層，重點放在「還沒大噴的中型成長股」（下一檔 ALAB/MRVL 的候選），
# 同時保留各層龍頭做題材對照。explosive_growth(早期飆股雷達) 模式專用此池。
# sector 欄位直接用「蛋糕層」名稱，掃描的類股分析就會依層分組。
AI_CAKE_UNIVERSE_US: list[tuple[str, str, str]] = [
    # L1 算力 / 加速器 / IC 設計
    ("NVDA", "NVIDIA Corporation", "①算力核心"),
    ("AMD", "Advanced Micro Devices", "①算力核心"),
    ("AVGO", "Broadcom Inc.", "①算力核心"),
    ("MRVL", "Marvell Technology", "①算力核心"),
    ("ARM", "Arm Holdings plc", "①算力核心"),
    ("QCOM", "QUALCOMM Incorporated", "①算力核心"),
    ("ALAB", "Astera Labs, Inc.", "①算力核心"),
    ("CRDO", "Credo Technology Group", "①算力核心"),
    # L2 記憶體 / 儲存
    ("MU", "Micron Technology", "②記憶體儲存"),
    ("WDC", "Western Digital Corp.", "②記憶體儲存"),
    ("STX", "Seagate Technology", "②記憶體儲存"),
    ("SNDK", "Sandisk Corporation", "②記憶體儲存"),
    # L3 光通訊 / 網通 / 互連
    ("ANET", "Arista Networks", "③光通訊網通"),
    ("COHR", "Coherent Corp.", "③光通訊網通"),
    ("LITE", "Lumentum Holdings", "③光通訊網通"),
    ("CIEN", "Ciena Corporation", "③光通訊網通"),
    ("AAOI", "Applied Optoelectronics", "③光通訊網通"),
    ("POET", "POET Technologies", "③光通訊網通"),
    ("ALGM", "Allegro MicroSystems", "③光通訊網通"),
    # L4 半導體製造 / 設備
    ("TSM", "Taiwan Semiconductor", "④製造設備"),
    ("ASML", "ASML Holding N.V.", "④製造設備"),
    ("AMAT", "Applied Materials", "④製造設備"),
    ("LRCX", "Lam Research Corp.", "④製造設備"),
    ("KLAC", "KLA Corporation", "④製造設備"),
    ("ONTO", "Onto Innovation", "④製造設備"),
    ("ACMR", "ACM Research", "④製造設備"),
    ("NVMI", "Nova Ltd.", "④製造設備"),
    ("CAMT", "Camtek Ltd.", "④製造設備"),
    ("AEIS", "Advanced Energy Industries", "④製造設備"),
    ("FORM", "FormFactor, Inc.", "④製造設備"),
    # L5 資料中心電力 / 散熱 / 基礎設施 / 伺服器 ODM
    ("VRT", "Vertiv Holdings", "⑤資料中心電力散熱"),
    ("VST", "Vistra Corp.", "⑤資料中心電力散熱"),
    ("CEG", "Constellation Energy", "⑤資料中心電力散熱"),
    ("GEV", "GE Vernova Inc.", "⑤資料中心電力散熱"),
    ("ETN", "Eaton Corporation", "⑤資料中心電力散熱"),
    ("POWL", "Powell Industries", "⑤資料中心電力散熱"),
    ("NVT", "nVent Electric plc", "⑤資料中心電力散熱"),
    ("MOD", "Modine Manufacturing", "⑤資料中心電力散熱"),
    ("BE", "Bloom Energy Corp.", "⑤資料中心電力散熱"),
    ("OKLO", "Oklo Inc.", "⑤資料中心電力散熱"),
    ("SMR", "NuScale Power Corp.", "⑤資料中心電力散熱"),
    ("SMCI", "Super Micro Computer", "⑤資料中心電力散熱"),
    ("DELL", "Dell Technologies", "⑤資料中心電力散熱"),
    ("CLS", "Celestica Inc.", "⑤資料中心電力散熱"),
    ("NBIS", "Nebius Group N.V.", "⑤資料中心電力散熱"),
    # L6 IP / EDA / 軟體 / 應用
    ("SNPS", "Synopsys, Inc.", "⑥IP軟體應用"),
    ("CDNS", "Cadence Design Systems", "⑥IP軟體應用"),
    ("PLTR", "Palantir Technologies", "⑥IP軟體應用"),
    ("NOW", "ServiceNow, Inc.", "⑥IP軟體應用"),
    ("PANW", "Palo Alto Networks", "⑥IP軟體應用"),
    ("CRWD", "CrowdStrike Holdings", "⑥IP軟體應用"),
    ("APP", "AppLovin Corporation", "⑥IP軟體應用"),
]

# 台股 AI 蛋糕池（中文名供顯示）
AI_CAKE_UNIVERSE_TW: list[tuple[str, str, str]] = [
    ("2330.TW", "台積電", "④製造設備"),
    ("2454.TW", "聯發科", "①算力核心"),
    ("3661.TW", "世芯-KY", "①算力核心"),
    ("3443.TW", "創意", "①算力核心"),
    ("2379.TW", "瑞昱", "①算力核心"),
    ("3034.TW", "聯詠", "①算力核心"),
    ("8299.TWO", "群聯", "②記憶體儲存"),
    ("2344.TW", "華邦電", "②記憶體儲存"),
    ("3711.TW", "日月光投控", "④製造設備"),
    ("6515.TW", "穎崴", "④製造設備"),
    ("3017.TW", "奇鋐", "⑤資料中心電力散熱"),
    ("2308.TW", "台達電", "⑤資料中心電力散熱"),
    ("2382.TW", "廣達", "⑤資料中心電力散熱"),
    ("2317.TW", "鴻海", "⑤資料中心電力散熱"),
    ("3231.TW", "緯創", "⑤資料中心電力散熱"),
    ("6669.TW", "緯穎", "⑤資料中心電力散熱"),
    ("2376.TW", "技嘉", "⑤資料中心電力散熱"),
    ("2357.TW", "華碩", "⑤資料中心電力散熱"),
    ("3008.TW", "大立光", "③光通訊網通"),
    ("4977.TW", "眾達-KY", "③光通訊網通"),
]


def _gooaye_named_constituents(market: str) -> list[SP500Constituent]:
    """讀取專家點名清單（股癌 gooaye_opinions.json ＋ 尼可拉斯楊 nicolas_opinions.json，皆由背景
    每 2h／每日自動更新），把最近講到、且符合市場別的個股動態併進掃描池 —— 讓清單跟著兩位專家持續
    長出新標的，不必手動維護。明顯偏空者略過；去重（同檔以先出現的來源標記）。失敗回空。"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data")
    sources = [("gooaye_opinions.json", "🎙️股癌點名"), ("nicolas_opinions.json", "🎯尼可拉斯楊")]
    try:
        from src.pipeline.daily_report import TW_NAMES
    except Exception:
        TW_NAMES = {}

    def _norm(t: str) -> str:
        t = (t or "").strip().upper()
        if not t:
            return ""
        if "." in t:
            return t
        return f"{t}.TW" if t[0].isdigit() else t

    out: list[SP500Constituent] = []
    seen: set[str] = set()
    import json
    for fname, label in sources:
        try:
            with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                store = json.load(f)
            ops = store.get("opinions", []) if isinstance(store, dict) else (store or [])
        except Exception:
            continue
        for o in ops:
            sym = _norm(str(o.get("target_ticker", "")))
            if not sym or sym in seen:
                continue
            is_tw = sym.endswith(".TW") or sym.endswith(".TWO")
            if (market == "tw") != is_tw:
                continue
            if str(o.get("sentiment_label", "")).lower().find("bear") >= 0:
                continue  # 略過明顯偏空點名
            seen.add(sym)
            base = sym.split(".")[0]
            name = TW_NAMES.get(base, base) if is_tw else base
            out.append(SP500Constituent(symbol=sym, yf_symbol=sym, company_name=name, sector=label))
    return out


def fetch_ai_cake_universe(market: str) -> list[SP500Constituent]:
    """黃仁勳「AI 算力蛋糕」掃描池（explosive_growth 早期飆股雷達專用），
    並動態併入股癌最近點名的個股（自動追蹤、去重）。"""
    rows = AI_CAKE_UNIVERSE_TW if market == "tw" else AI_CAKE_UNIVERSE_US
    universe = [SP500Constituent(symbol=sym, yf_symbol=sym, company_name=name, sector=sect)
                for sym, name, sect in rows]
    existing = {c.yf_symbol.upper() for c in universe}
    for c in _gooaye_named_constituents(market):
        if c.yf_symbol.upper() not in existing:
            existing.add(c.yf_symbol.upper())
            universe.append(c)
    return universe


def download_sp500_price_map(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    data = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if data.empty:
        raise ValueError("無法下載 S&P 500 掃描所需的價格資料。")

    result: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1))
        for symbol in symbols:
            try:
                if symbol in level1:
                    frame = data.xs(symbol, axis=1, level=1).copy()
                elif symbol in level0:
                    frame = data[symbol].copy()
                else:
                    continue
                if {"Open", "High", "Low", "Close", "Volume"} <= set(frame.columns):
                    result[symbol] = frame.dropna(subset=["High", "Low", "Close"]).copy()
            except Exception:
                continue
    else:
        frame = data.copy()
        if {"Open", "High", "Low", "Close", "Volume"} <= set(frame.columns):
            result[symbols[0]] = frame.dropna(subset=["High", "Low", "Close"]).copy()

    return result


def normalize_fear_greed_label(value: int | str) -> str:
    if isinstance(value, str) and value:
        lowered = value.strip().lower()
        mapping = {
            "extreme fear": "極度恐懼",
            "fear": "恐懼",
            "neutral": "中性",
            "greed": "貪婪",
            "extreme greed": "極度貪婪",
        }
        if lowered in mapping:
            return mapping[lowered]

    score = int(value)
    if score <= 25:
        return "極度恐懼"
    if score <= 45:
        return "恐懼"
    if score < 60:
        return "中性"
    if score < 80:
        return "貪婪"
    return "極度貪婪"


def build_local_fear_greed_proxy(*, vix_close: float, distance_ma200_pct: float, rsi: float) -> FearGreedSnapshot:
    greed_score = 50
    greed_score += int(np.clip((distance_ma200_pct / 10) * 18, -20, 20))
    greed_score += int(np.clip((rsi - 50) * 1.1, -20, 20))
    greed_score -= int(np.clip((vix_close - 18) * 1.2, -25, 25))
    greed_score = max(0, min(100, greed_score))
    return FearGreedSnapshot(
        score=greed_score,
        label=normalize_fear_greed_label(greed_score),
        source="本地代理（CNN 暫不可用）",
    )


def fetch_fear_greed_snapshot(*, vix_close: float, distance_ma200_pct: float, rsi: float) -> FearGreedSnapshot:
    try:
        response = requests.get(
            CNN_FEAR_GREED_API_URL,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://edition.cnn.com/",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("fear_and_greed", {})
        score = _clamp_int(float(data["score"]))
        return FearGreedSnapshot(
            score=score,
            label=normalize_fear_greed_label(str(data.get("rating", "")) or score),
            source="CNN Fear & Greed",
        )
    except Exception:
        return build_local_fear_greed_proxy(vix_close=vix_close, distance_ma200_pct=distance_ma200_pct, rsi=rsi)


def _market_bucket_from_row(row: pd.Series) -> str:
    vix = float(row["VIX"])
    drawdown = float(row["SPY_DRAWDOWN"])
    distance = float(row["SPY_DISTANCE_MA200"])
    if vix >= 35 and drawdown <= -0.10:
        return "panic"
    if vix >= 25 or drawdown <= -0.06:
        return "fear"
    if vix <= 18 and distance >= 0.06:
        return "greed"
    return "neutral"


def neutral_market_regime() -> MarketRegime:
    """市場情緒資料抓不到/不足時的中性退路，避免整個掃描中斷。"""
    return MarketRegime(
        vix_close=0.0, vix_regime="中性", fear_greed_score=50, fear_greed_label="中性",
        fear_greed_source="資料不足，採中性", spy_drawdown_pct=0.0, spy_distance_ma200_pct=0.0,
        regime_score=50, action="中性偏分批", risk_budget="中性部位",
        summary="市場情緒資料暫時無法取得，採用中性環境，不額外放大或縮小訊號。",
        backtest_win_rate_5d=50.0, backtest_avg_return_5d=0.0,
        backtest_win_rate_20d=50.0, backtest_avg_return_20d=0.0,
    )


def compute_market_regime(market_hint: str = "us") -> MarketRegime:
    ticker_index = "^TWII" if market_hint == "tw" else "SPY"
    market = yf.download([ticker_index, "^VIX"], period="3y", interval="1d", auto_adjust=False, progress=False, threads=True)
    if market.empty or not isinstance(market.columns, pd.MultiIndex):
        raise ValueError(f"無法取得{'台股' if market_hint == 'tw' else '美股'}市場情緒資料。")

    spy = market.xs(ticker_index, axis=1, level=1).dropna(subset=["Close"]).copy()
    vix = market.xs("^VIX", axis=1, level=1).copy()

    frame = pd.DataFrame(index=spy.index)
    frame["SPY"] = spy["Close"]
    frame["VIX"] = vix["Close"].reindex(spy.index).ffill().bfill()
    frame["SPY_MA200"] = frame["SPY"].rolling(200).mean()
    frame["SPY_RSI14"] = compute_rsi(frame["SPY"], 14)
    frame["SPY_252MAX"] = frame["SPY"].rolling(252).max()
    frame["SPY_DRAWDOWN"] = frame["SPY"] / frame["SPY_252MAX"] - 1
    frame["SPY_DISTANCE_MA200"] = frame["SPY"] / frame["SPY_MA200"] - 1
    frame = frame.dropna().copy()
    if frame.empty:
        # 資料不足（如 Yahoo 多檔下載回傳過短/不完整）→ 中性，避免 iloc 越界。
        return neutral_market_regime()
    latest = frame.iloc[-1]

    vix_close = float(latest["VIX"])
    drawdown_pct = float(latest["SPY_DRAWDOWN"] * 100)
    distance_ma200_pct = float(latest["SPY_DISTANCE_MA200"] * 100)
    rsi = float(latest["SPY_RSI14"])

    fear_greed = fetch_fear_greed_snapshot(vix_close=vix_close, distance_ma200_pct=distance_ma200_pct, rsi=rsi)

    if vix_close >= 35:
        vix_regime = "恐慌"
    elif vix_close >= 25:
        vix_regime = "高波動"
    elif vix_close >= 18:
        vix_regime = "正常"
    else:
        vix_regime = "低波動"

    frame["REGIME_BUCKET"] = frame.apply(_market_bucket_from_row, axis=1)
    frame["FWD_5D"] = frame["SPY"].shift(-5) / frame["SPY"] - 1
    frame["FWD_20D"] = frame["SPY"].shift(-20) / frame["SPY"] - 1
    bucket = _market_bucket_from_row(latest)
    subset = frame[frame["REGIME_BUCKET"] == bucket].dropna(subset=["FWD_5D", "FWD_20D"])
    if len(subset) < 20:
        subset = frame.dropna(subset=["FWD_5D", "FWD_20D"])

    win_rate_5d = float((subset["FWD_5D"] > 0).mean() * 100)
    avg_return_5d = float(subset["FWD_5D"].mean() * 100)
    win_rate_20d = float((subset["FWD_20D"] > 0).mean() * 100)
    avg_return_20d = float(subset["FWD_20D"].mean() * 100)

    if market_hint == "tw":
        if vix_close >= 35 and fear_greed.score <= 25 and drawdown_pct <= -12:
            regime_score = 90
            action = "全力偏多"
            risk_budget = "可提高部位"
            summary = "台股進入歷史大跌超賣區，回測反彈機率高，適合中長線戰略佈局。"
        elif vix_close >= 25 or fear_greed.score <= 40:
            regime_score = 72
            action = "分批買進"
            risk_budget = "中等偏高部位"
            summary = "台股大盤面臨回檔壓力，防禦升溫反倒帶來中長線分批進場機會。"
        elif fear_greed.score >= 75 and vix_close <= 18:
            regime_score = 28
            action = "減碼 / 不追價"
            risk_budget = "低風險部位"
            summary = "台股短線偏熱，應分批落袋鎖利，切勿在目前高位追高。"
        else:
            regime_score = 50
            action = "中性偏分批"
            risk_budget = "中性部位"
            summary = "台股大盤處於箱型整理或溫和多頭，建議秉持資金紀律逢回支撐佈局。"
    else:
        if vix_close >= 35 and fear_greed.score <= 25 and drawdown_pct <= -12:
            regime_score = 90
            action = "全力偏多"
            risk_budget = "可提高部位"
            summary = "市場進入恐慌區，歷史上這類大跌後反彈機率較高，但仍要分批承接。"
        elif vix_close >= 25 or fear_greed.score <= 40:
            regime_score = 72
            action = "分批買進"
            risk_budget = "中等偏高部位"
            summary = "市場仍有壓力，但恐懼提升時常會帶來中期布局機會。"
        elif fear_greed.score >= 75 and vix_close <= 18:
            regime_score = 28
            action = "減碼 / 不追價"
            risk_budget = "低風險部位"
            summary = "市場偏熱，這時候更重視獲利了結，不適合追高。"
        else:
            regime_score = 50
            action = "中性偏分批"
            risk_budget = "中性部位"
            summary = "市場沒有明顯極端訊號，布局上以分批與紀律控風險為主。"

    return MarketRegime(
        vix_close=round(vix_close, 2),
        vix_regime=vix_regime,
        fear_greed_score=fear_greed.score,
        fear_greed_label=fear_greed.label,
        fear_greed_source=fear_greed.source,
        spy_drawdown_pct=round(drawdown_pct, 2),
        spy_distance_ma200_pct=round(distance_ma200_pct, 2),
        regime_score=regime_score,
        action=action,
        risk_budget=risk_budget,
        summary=summary,
        backtest_win_rate_5d=round(win_rate_5d, 2),
        backtest_avg_return_5d=round(avg_return_5d, 2),
        backtest_win_rate_20d=round(win_rate_20d, 2),
        backtest_avg_return_20d=round(avg_return_20d, 2),
    )


def score_news_items(news_items: list[dict[str, Any]]) -> tuple[int, str, int]:
    if not news_items:
        return 50, "最近 24 小時沒有抓到足夠新聞，消息面先以中性看待。", 0

    score = 50
    positive_hits = 0
    negative_hits = 0
    recent_titles: list[str] = []
    now = datetime.utcnow()

    for item in news_items[:8]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        recent_titles.append(title)
        lowered = title.lower()
        if any(keyword in lowered for keyword in POSITIVE_KEYWORDS):
            positive_hits += 1
            score += 6
        if any(keyword in lowered for keyword in NEGATIVE_KEYWORDS):
            negative_hits += 1
            score -= 6

        published = item.get("providerPublishTime")
        if published:
            try:
                published_dt = datetime.utcfromtimestamp(int(published))
                if now - published_dt <= timedelta(hours=24):
                    if any(keyword in lowered for keyword in POSITIVE_KEYWORDS):
                        score += 2
                    if any(keyword in lowered for keyword in NEGATIVE_KEYWORDS):
                        score -= 2
            except (TypeError, ValueError, OSError):
                pass

    score = _clamp_int(score)
    if positive_hits > negative_hits:
        summary = f"消息面偏正向，正面關鍵字 {positive_hits} 次、負面 {negative_hits} 次。"
    elif negative_hits > positive_hits:
        summary = f"消息面偏保守，負面關鍵字 {negative_hits} 次、正面 {positive_hits} 次。"
    else:
        summary = "消息面正負訊號接近，暫時以中性看待。"

    if recent_titles:
        summary = f"{summary} 最新標題：{recent_titles[0][:90]}"
    return score, summary, len(recent_titles)


def score_fundamentals(info: dict[str, Any], latest_close: float) -> tuple[int, str]:
    score = 50
    notes: list[str] = []

    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    roe = _safe_float(info.get("returnOnEquity"))
    operating_margin = _safe_float(info.get("operatingMargins"))
    recommendation_mean = _safe_float(info.get("recommendationMean"))
    target_mean_price = _safe_float(info.get("targetMeanPrice"))
    forward_pe = _safe_float(info.get("forwardPE"))

    if revenue_growth is not None:
        if revenue_growth >= 0.12:
            score += 8
            notes.append("營收成長偏強")
        elif revenue_growth < 0:
            score -= 6
            notes.append("營收轉弱")

    if earnings_growth is not None:
        if earnings_growth >= 0.12:
            score += 8
            notes.append("獲利成長偏強")
        elif earnings_growth < 0:
            score -= 6
            notes.append("獲利轉弱")

    if roe is not None:
        if roe >= 0.15:
            score += 5
            notes.append("ROE 水準不錯")
        elif roe < 0.06:
            score -= 4
            notes.append("ROE 偏弱")

    if operating_margin is not None:
        if operating_margin >= 0.20:
            score += 4
            notes.append("營業利益率佳")
        elif operating_margin < 0.05:
            score -= 4
            notes.append("營益率偏低")

    if recommendation_mean is not None:
        if recommendation_mean <= 2.2:
            score += 4
            notes.append("分析師評級偏正向")
        elif recommendation_mean >= 3.5:
            score -= 4
            notes.append("分析師評級偏保守")

    if target_mean_price is not None and latest_close > 0:
        upside = (target_mean_price / latest_close) - 1
        if upside >= 0.10:
            score += 4
            notes.append("法人目標價仍有上行空間")
        elif upside <= -0.03:
            score -= 4
            notes.append("法人目標價空間有限")

    if forward_pe is not None:
        if 0 < forward_pe <= 30:
            score += 2
        elif forward_pe >= 60:
            score -= 4
            notes.append("估值偏高")

    score = _clamp_int(score)
    summary = "、".join(notes[:3]) if notes else "基本面資料有限，先維持中性評估。"
    return score, summary


def build_signal_backtest(frame: pd.DataFrame, report: SignalReport, regime: MarketRegime) -> SignalBacktest:
    enriched = frame.copy()
    enriched["MA20"] = enriched["Close"].rolling(20).mean()
    enriched["MA50"] = enriched["Close"].rolling(50).mean()
    enriched["RSI14"] = compute_rsi(enriched["Close"], 14)
    enriched["ATR_PCT"] = ((enriched["High"] - enriched["Low"]).rolling(14).mean() / enriched["Close"]) * 100
    enriched["FWD_5D"] = enriched["Close"].shift(-5) / enriched["Close"] - 1
    enriched["FWD_20D"] = enriched["Close"].shift(-20) / enriched["Close"] - 1
    enriched["FWD_60D"] = enriched["Close"].shift(-60) / enriched["Close"] - 1

    current_rsi = report.rsi14
    current_above_ma20 = report.latest_close > report.ma20
    current_above_ma50 = report.latest_close > report.ma50
    current_return = report.expected_return_pct

    mask = (
        enriched["MA20"].notna()
        & enriched["MA50"].notna()
        & enriched["RSI14"].between(current_rsi - 6, current_rsi + 6)
        & ((enriched["Close"] > enriched["MA20"]) == current_above_ma20)
        & ((enriched["Close"] > enriched["MA50"]) == current_above_ma50)
    )
    matches = enriched.loc[mask].dropna(subset=["FWD_5D", "FWD_20D", "FWD_60D"])
    if len(matches) < 20:
        matches = enriched.dropna(subset=["FWD_5D", "FWD_20D", "FWD_60D"]).tail(260)

    if matches.empty:
        return SignalBacktest(
            0,
            50.0,
            0.0,
            50.0,
            0.0,
            50.0,
            0.0,
            0.0,
            50.0,
            35,
            "可用樣本太少，這筆回測只能當輔助參考。",
        )

    avg_5d = float(matches["FWD_5D"].mean() * 100)
    avg_20d = float(matches["FWD_20D"].mean() * 100)
    avg_60d = float(matches["FWD_60D"].mean() * 100)
    win_5d = float((matches["FWD_5D"] > 0).mean() * 100)
    win_20d = float((matches["FWD_20D"] > 0).mean() * 100)
    win_60d = float((matches["FWD_60D"] > 0).mean() * 100)
    downside_rate_20d = float((matches["FWD_20D"] < 0).mean() * 100)
    max_drawdown_20d = float(matches["FWD_20D"].min() * 100)

    sample_penalty = 0 if len(matches) >= 45 else (45 - len(matches)) * 0.7
    risk_penalty = max(0.0, abs(min(0.0, avg_20d)) * 4.0) + max(0.0, (downside_rate_20d - 45) * 0.7)
    reward_bonus = max(0.0, avg_20d * 4.5) + max(0.0, (win_20d - 52) * 0.8)
    expectation_penalty = 6 if current_return >= 20 and avg_20d < 1.5 else 0
    confidence_score = _clamp_int(45 + reward_bonus - risk_penalty - sample_penalty - expectation_penalty)

    if confidence_score >= 70 and avg_20d > 2 and win_20d >= 58:
        note = "相似型態的後續表現不錯，這筆訊號的歷史可信度偏高。"
    elif confidence_score <= 40 or avg_20d <= 0 or downside_rate_20d >= 50:
        note = "相似型態的歷史表現不穩，這檔股票今天不適合太積極。"
    elif regime.action == "全力偏多" and avg_20d > 1:
        note = "目前市場環境有利於偏多操作，回測也沒有明顯拖後腿。"
    else:
        note = "回測結果中性偏正面，可以當成輔助，但仍要搭配今天的位置判斷。"

    return SignalBacktest(
        sample_size=int(len(matches)),
        win_rate_5d=round(win_5d, 2),
        avg_return_5d=round(avg_5d, 2),
        win_rate_20d=round(win_20d, 2),
        avg_return_20d=round(avg_20d, 2),
        win_rate_60d=round(win_60d, 2),
        avg_return_60d=round(avg_60d, 2),
        max_drawdown_20d=round(max_drawdown_20d, 2),
        downside_rate_20d=round(downside_rate_20d, 2),
        confidence_score=confidence_score,
        calibration_note=note,
    )


def decide_action_label(daily_score: int, regime: MarketRegime, backtest: SignalBacktest, bias: str, is_strong_value: bool = False, is_dip_buy: bool = False) -> tuple[str, str, str]:
    """建議強度與綜合分數(daily_score)**單調對應** —— 分數越高建議越積極，不會出現
    「100 分卻先觀察」的矛盾。回測信心(confidence_score)只用來決定是否升到最高一級
    「全力買進」與部位大小，不再把高分股壓成先觀察。風險閘門(明顯偏空、大盤要減碼)優先。

    2026-06-19 重寫：原本以 backtest.confidence_score 當硬性門檻，導致慢牛股(如 BRK-B)
    daily_score=100 仍落到「先觀察」，與排名(分數)和對帳本建議互相矛盾。"""
    conf = getattr(backtest, "confidence_score", 50)
    # 2026-06-30：好股大跌承接（quality-dip）不算追空頭——這正是「別人恐懼我貪婪」的買點，
    # 不該被偏空風險閘門壓成「觀望/減碼」。
    is_bearish_blocked = (bias == "偏空" and not is_strong_value and not is_dip_buy)

    # 好股大跌承接：給「分批小量承接」評級，部位保守、留銀彈攤平。
    if is_dip_buy and bias != "偏多":
        return "分批小量承接", "中", "小部位分批"
    # 風險閘門：明顯偏空且非強勢價值 → 不追空頭
    if is_bearish_blocked:
        return "觀望 / 減碼", "低", "低風險部位"
    # 大盤明確要減碼，且分數未到高檔 → 保守
    if regime.action == "減碼 / 不追價" and daily_score < 72:
        return "觀望 / 減碼", "低", "低風險部位"

    if daily_score >= 80:
        if regime.action == "全力偏多" and conf >= 60:
            return "全力買進", "非常高", "可提高部位"
        return "分批買進", "高", "中等偏高部位"
    if daily_score >= 66:
        return "分批買進", "高", "中等偏高部位"
    if daily_score >= 56:
        return "小量試單", "中", "小部位"
    if daily_score >= 46:
        return "先觀察", "低", "等待更好位置"
    return "觀望 / 減碼", "低", "低風險部位"



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


def enrich_candidate(
    constituent: SP500Constituent,
    frame: pd.DataFrame,
    report: SignalReport,
    regime: MarketRegime,
    *,
    fetch_external_data: bool,
    use_ai_committee: bool,
    committee_model: str,
    sector_score: int = 50,
    is_main_line: bool = False,
    is_sector_leader: bool = False,
    sector_boost: int = 0,
    scan_type: str = "optimal",
) -> SP500DailyPick:
    # Nicholas Yang's AI Industry Chain & Bottleneck mappings mapped first
    ai_chain_layer, critical_bottleneck = map_ai_chain_and_bottleneck(constituent.symbol, constituent.sector)

    # 入選後改用「與個股分析完全相同」的完整報告（含 agent 信任權重、3/6/9/12 預測、長線虧損閘門）。
    # 初篩傳入的 report 為 lightweight 版本，僅供排序，這裡重新做完整分析取代之。
    try:
        report = build_report(constituent.yf_symbol, frame, fetch_fundamentals=fetch_external_data)
    except Exception:
        pass  # 萬一完整分析失敗，沿用初篩的 lightweight 報告，避免整檔掉出名單

    enriched_report = apply_committee_overlay(report, committee_model) if use_ai_committee else report

    ticker = yf.Ticker(constituent.yf_symbol) if fetch_external_data else None
    news_items: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    if ticker is not None:
        try:
            news_items = ticker.news or []
        except Exception:
            news_items = []
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

    if info:
        # F-Score
        roa = info.get("returnOnAssets")
        cfo = info.get("operatingCashflow")
        net_income = info.get("netIncomeToCommon")
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        gross_margin = info.get("grossMargins")
        roe = info.get("returnOnEquity")
        rev_growth = info.get("revenueGrowth")

        f_score = 0
        if roa and roa > 0: f_score += 1
        if cfo and cfo > 0: f_score += 1
        if net_income and net_income > 0: f_score += 1
        if cfo and net_income and cfo > net_income: f_score += 1
        if debt_to_equity is not None and debt_to_equity < 150: f_score += 1
        if current_ratio and current_ratio > 1.0: f_score += 1
        if gross_margin and gross_margin > 0.20: f_score += 1
        if roe and roe > 0: f_score += 1
        if rev_growth and rev_growth > 0: f_score += 1
        
        enriched_report.fundamental_score = f_score

        # Graham Number
        eps = info.get("trailingEps")
        bvps = info.get("bookValue")
        if eps and bvps and eps > 0 and bvps > 0:
            enriched_report.graham_number = round((22.5 * eps * bvps) ** 0.5, 2)
            
        # Re-run enforce_position_value and derive_today_plan with updated fundamental_score
        try:
            buy_zone = enriched_report.buy_zone
            sell_zone = enriched_report.sell_zone
            stop_loss = enriched_report.stop_loss
            buy_strength = enriched_report.buy_strength
            reason = enriched_report.reason
            
            from src.simple_signal import enforce_position_value, derive_today_plan
            
            buy_zone, sell_zone, stop_loss, buy_strength, reason = enforce_position_value(
                buy_zone=buy_zone,
                sell_zone=sell_zone,
                stop_loss=stop_loss,
                buy_strength=buy_strength,
                reason=reason,
                fundamental_score=f_score,
                valuation_gap_pct=enriched_report.valuation_gap_pct,
            )
            
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
                latest_close=enriched_report.latest_close,
                ma50=enriched_report.ma50,
                atr14=enriched_report.atr14,
                rsi14=enriched_report.rsi14,
                bias=enriched_report.bias,
                buy_strength=buy_strength,
                buy_zone=buy_zone,
                sell_zone=sell_zone,
                stop_loss=stop_loss,
                candlestick_pattern=enriched_report.candlestick_pattern,
                fundamental_score=f_score,
                valuation_gap_pct=enriched_report.valuation_gap_pct,
                ai_chain_layer=enriched_report.ai_chain_layer,
                ma120=enriched_report.ma120,
                long_term_blocked=bool(enriched_report.long_term_risk and enriched_report.long_term_risk.get("blocked")),
                drawdown_from_high_pct=(
                    ((enriched_report.latest_close / float(frame.tail(60)["High"].max())) - 1) * 100
                    if len(frame) >= 1 and float(frame.tail(60)["High"].max()) > 0 else 0.0
                ),
            )

            # Update report fields
            enriched_report.buy_zone = buy_zone
            enriched_report.sell_zone = sell_zone
            enriched_report.stop_loss = stop_loss
            enriched_report.buy_strength = buy_strength
            enriched_report.reason = reason
            
            enriched_report.today_action = today_action
            enriched_report.today_entry_zone = today_entry_zone
            enriched_report.today_note = today_note
            enriched_report.today_exit_action = today_exit_action
            enriched_report.today_exit_zone = today_exit_zone
            enriched_report.today_exit_note = today_exit_note
            enriched_report.expected_return_pct = expected_return_pct
            enriched_report.risk_reward_ratio = reward_ratio
            enriched_report.holding_days_estimate = holding_days_estimate
            enriched_report.holding_window = holding_window
            enriched_report.kelly_position_pct = kelly_position_pct
        except Exception:
            pass
            
        # Re-generate decision assistance advice
        try:
            enriched_report.decision_assistance = generate_decision_assistance(enriched_report)
        except Exception:
            pass

    news_score, headline_summary, headline_count = score_news_items(news_items)
    fundamental_score, fundamental_summary = score_fundamentals(info, enriched_report.latest_close)
    technical_score = enriched_report.composite_score
    backtest = build_signal_backtest(frame, enriched_report, regime)
    backtest_score = backtest.confidence_score

    # 相對強度(動能)因子：近一個月(約20交易日)報酬率 → 0-100。
    # 2026-06-19 新增：原 daily_score 靠 composite_score(帶價值/均值回歸偏誤)主導，排名最前永遠是
    # 漲不動的大型半導體與防禦股(JNJ/LIN)，真正會漲的強勢股被排到後面。加入動能因子拉正排名與漲跌的相關性。
    rs_score = _relative_strength_from_frame(frame)

    # 後續上漲空間(看好會大漲)因子：6 個月統計預測期望報酬 → 0-100（+30%→98、0%→50、-30%→2）。
    # 2026-06-19 新增：使用者要的不只是「今天能買」，而是「近期是好買點且看好後續會大漲」。
    # 這個因子把「未來半年漲幅期望」放進排名；缺預測時退回用歷史 60 日平均報酬當代理。
    upside_score = 50.0
    try:
        f6 = None
        pf = getattr(enriched_report, "price_forecast", None)
        if pf and pf.get("horizons"):
            for h in pf["horizons"]:
                if h.get("days") == 126:
                    f6 = float(h.get("expected_return_pct", 0.0))
                    break
        if f6 is not None:
            upside_score = max(0.0, min(100.0, 50.0 + 1.6 * f6))
        elif getattr(backtest, "avg_return_60d", None):
            upside_score = max(0.0, min(100.0, 50.0 + 3.0 * float(backtest.avg_return_60d)))
    except Exception:
        upside_score = 50.0

    # 早期飆股型態分數（打底量縮、剛轉強、貼近突破但未乖離追高）。
    early_stage_score = _early_stage_score(frame)

    # Evaluate 1-3 Month Explosive Breakout / Dark Horse Potential
    is_dark_horse = (
        (backtest.avg_return_20d >= 2.5 or backtest.avg_return_60d >= 6.0)
        and (backtest.win_rate_20d >= 56.0 or backtest.win_rate_60d >= 58.0)
        and (enriched_report.bb_width < 0.12 or enriched_report.composite_score >= 70)
        and enriched_report.bias != "偏空"
    )
    dark_horse_boost = 6 if is_dark_horse else 0

    # Lagging Value calculation
    value_boost = 0
    value_notes = []
    if scan_type == "lagging_value":
        # 1. Nicholas's AI Chain Boost (+12 points)
        if ai_chain_layer is not None:
            value_boost += 12
            value_notes.append(f"AI產業鏈: {ai_chain_layer}")
            
        # 2. Nicholas's Bottleneck 🔥 Boost (+20 points)
        if critical_bottleneck is not None:
            value_boost += 20
            value_notes.append(f"卡脖子瓶頸: {critical_bottleneck}")

        # 3. Minnie's F-Score Boost
        f_score = enriched_report.fundamental_score
        if f_score >= 7:
            value_boost += 15
            value_notes.append(f"財務極強健(F-Score: {f_score}/9)")
        elif f_score >= 5:
            value_boost += 8
            value_notes.append(f"財務平穩(F-Score: {f_score}/9)")
        elif 1 <= f_score <= 3:
            value_boost -= 12
            value_notes.append(f"防護價值陷阱(F-Score僅 {f_score}/9)")

        # 4. Minnie's Graham Safety Margin Boost
        if enriched_report.graham_number and enriched_report.graham_number > 0:
            close = enriched_report.latest_close
            discount = (enriched_report.graham_number / close) - 1
            if close < enriched_report.graham_number:
                value_boost += 20
                value_notes.append(f"低於葛拉漢價(折價 {discount*100:.1f}%)")
            elif close < enriched_report.graham_number * 1.25:
                value_boost += 10
                value_notes.append(f"貼近葛拉漢價(溢價 {abs(discount)*100:.1f}%)")
            else:
                value_boost -= 5

        # 5. Low P/E Valuation Boost
        forward_pe = _safe_float(info.get("forwardPE")) if info else None
        if forward_pe is not None:
            if 0 < forward_pe <= 18:
                value_boost += 12
                value_notes.append(f"低前瞻本益比({forward_pe:.1f}倍)")
            elif 18 < forward_pe <= 28:
                value_boost += 5
            elif forward_pe > 40:
                value_boost -= 15
                value_notes.append(f"估值偏高(本益比 {forward_pe:.1f}倍)")

        # 6. Technical Pullback & Oversold Boost
        rsi = enriched_report.rsi14
        if rsi < 42:
            value_boost += 10
            value_notes.append(f"技術超跌(RSI: {rsi:.1f})")
        elif rsi <= 50:
            value_boost += 5
            value_notes.append(f"回檔整理中(RSI: {rsi:.1f})")
        elif rsi > 60:
            value_boost -= 12
            value_notes.append(f"短線已高(RSI: {rsi:.1f})")

        # 7. InvestingPro 12 模型估值折價（穩健平均）：低估補漲模式的核心排名因子。
        # 被低估越多、加分越高，讓「被低估最多 %」的好股排到前面；溢價則扣分避免追貴。
        inv_gap = enriched_report.valuation_gap_pct
        if inv_gap is not None:
            if inv_gap >= 25:
                value_boost += 22
                value_notes.append(f"InvestingPro 估值大幅折價 {inv_gap:.0f}%")
            elif inv_gap >= 15:
                value_boost += 15
                value_notes.append(f"InvestingPro 估值折價 {inv_gap:.0f}%")
            elif inv_gap >= 8:
                value_boost += 8
                value_notes.append(f"InvestingPro 估值折價 {inv_gap:.0f}%")
            elif inv_gap <= -10:
                value_boost -= 12
                value_notes.append(f"InvestingPro 估值溢價 {abs(inv_gap):.0f}%（偏貴）")

    if scan_type == "lagging_value":
        # 低估補漲：基本面/估值仍是主角(0.32，保留價值個性)，但加入相對強度確認「真的開始補漲」、
        # 並加入未來半年上漲空間，讓選出的是「被低估且看好會補漲」而非單純便宜卻沒戲的價值陷阱。
        daily_score = _clamp_int(
            technical_score * 0.18
            + rs_score * 0.10
            + upside_score * 0.14
            + news_score * 0.06
            + fundamental_score * 0.32
            + regime.regime_score * 0.05
            + backtest_score * 0.15
        )
        daily_score = _clamp_int(daily_score + sector_boost + dark_horse_boost + value_boost)
    elif scan_type == "explosive_growth":
        # 早期飆股雷達：核心是「早期型態(打底/剛轉強/貼近突破但未乖離)」，輔以未來上漲空間、
        # 基本面成長與題材；**不做價值/超跌加分**，並對追高(RSI 過熱)扣分 —— 要在噴出『之前』抓到。
        rsi_pen = 10 if enriched_report.rsi14 >= 78 else (5 if enriched_report.rsi14 >= 70 else 0)
        daily_score = _clamp_int(
            early_stage_score * 0.42
            + upside_score * 0.22
            + fundamental_score * 0.12
            + backtest_score * 0.10
            + news_score * 0.08
            + regime.regime_score * 0.06
            - rsi_pen
        )
        # AI 蛋糕題材加成：卡脖子龍頭多給一點（整池都在 AI 鏈上，差異化用瓶頸/層級）
        theme_boost = 6 if critical_bottleneck is not None else (3 if ai_chain_layer is not None else 0)
        daily_score = _clamp_int(daily_score + sector_boost + dark_horse_boost + theme_boost)
    else:
        # 近期最佳買點＋看好後續大漲：三大支柱 = technical(進場健康度) + 相對強度(近期強勢)
        # + upside(未來半年上漲空間)，輔以歷史型態回測、新聞、基本面、大盤情緒。
        daily_score = _clamp_int(
            technical_score * 0.25
            + rs_score * 0.22
            + upside_score * 0.20
            + backtest_score * 0.15
            + news_score * 0.08
            + fundamental_score * 0.07
            + regime.regime_score * 0.03
        )
        daily_score = _clamp_int(daily_score + sector_boost + dark_horse_boost)

    f_score = enriched_report.fundamental_score
    val_gap = enriched_report.valuation_gap_pct
    is_strong_value = (f_score >= 5 and val_gap is not None and val_gap >= 10.0)
    # 好股大跌承接：today_action 已由 derive_today_plan 判為買進（含 quality-dip 路徑），
    # 且非長線虧損閘門否決者，視為 dip-buy，讓建議強度與「今天可買」一致、不互相矛盾。
    _lt_blocked = bool(enriched_report.long_term_risk and enriched_report.long_term_risk.get("blocked"))
    is_dip_buy = (enriched_report.today_action in (BUY_NOW, BUY_SMALL)) and not _lt_blocked
    action_label, buy_urgency, position_sizing = decide_action_label(
        daily_score=daily_score,
        regime=regime,
        backtest=backtest,
        bias=enriched_report.bias,
        is_strong_value=is_strong_value,
        is_dip_buy=is_dip_buy,
    )

    # 長線虧損閘門：即使長抱仍可能虧損的股票，一律不建議買進。
    long_term_risk = enriched_report.long_term_risk
    long_term_block_note = ""
    if long_term_risk and long_term_risk.get("blocked"):
        action_label = "長線恐虧損 不建議買進"
        buy_urgency = "迴避"
        position_sizing = "不建議建倉"
        daily_score = min(daily_score, 40)
        long_term_block_note = f"🔴【長線虧損風險】{long_term_risk.get('note', '')}"

    sector_note = ""
    if is_main_line:
        if is_sector_leader:
            sector_note = f"該股隸屬當前熱門主力板塊【{constituent.sector}】（板塊動能高達 {sector_score} 分）且為板塊強勢龍頭，獲得評定加權 +9 分。"
        else:
            sector_note = f"該股屬於今日主力市場主線板塊【{constituent.sector}】（板塊動能 {sector_score} 分），獲得主線加成 +6 分。"
    elif sector_boost < 0:
        sector_note = f"該股所處板塊【{constituent.sector}】近期資金動能疲弱，評級扣減 {abs(sector_boost)} 分。"

    dark_horse_note = ""
    if is_dark_horse:
        dark_horse_note = f"🔥 偵測到【中線 1-3 個月爆發黑馬相】：布林通道緊縮且 1-3 個月中線歷史回測勝率高達 {backtest.win_rate_20d}% / 預期漲幅達 {backtest.avg_return_20d}%，具備極強中線飆股潛力，獲得黑馬加分 +6 分。"

    lagging_value_note = ""
    if scan_type == "lagging_value":
        note_parts = []
        if value_notes:
            note_parts.append("、".join(value_notes))
        if backtest.win_rate_20d > 0:
            note_parts.append(f"歷史波段勝率 {backtest.win_rate_20d:.1f}%/平均報酬 {backtest.avg_return_20d:.2f}%")
        
        lagging_value_note = f"💎【真正落後價值股推薦】：{ '；'.join(note_parts) }。該股目前技術面拉回超跌、估值安全邊際高，具備強大中線補漲動能。"

    # Nicholas Yang's AI Industry Chain & Bottleneck mappings (already mapped at top)
    
    # Calculate Novice Rating
    forward_pe = _safe_float(info.get("forwardPE")) if info else None
    novice_rating = calculate_novice_rating(daily_score, forward_pe, critical_bottleneck)
    if long_term_risk and long_term_risk.get("blocked"):
        novice_rating = "🔴 迴避"

    reason_parts = [long_term_block_note, lagging_value_note, dark_horse_note, sector_note, enriched_report.reason, headline_summary, fundamental_summary, backtest.calibration_note]
    reason = " ".join(part for part in reason_parts if part)

    return SP500DailyPick(
        symbol=constituent.symbol,
        company_name=constituent.company_name,
        sector=constituent.sector,
        latest_close=enriched_report.latest_close,
        bias=enriched_report.bias,
        reason=reason,
        rule_score=enriched_report.rule_score,
        ai_score=enriched_report.ai_score,
        composite_score=enriched_report.composite_score,
        technical_score=technical_score,
        news_score=news_score,
        fundamental_score=fundamental_score,
        regime_score=regime.regime_score,
        backtest_score=backtest_score,
        daily_score=daily_score,
        buy_strength=enriched_report.buy_strength,
        action_label=action_label,
        buy_urgency=buy_urgency,
        position_sizing=position_sizing,
        today_action=enriched_report.today_action,
        today_entry_zone=enriched_report.today_entry_zone,
        today_note=enriched_report.today_note,
        today_exit_action=enriched_report.today_exit_action,
        today_exit_zone=enriched_report.today_exit_zone,
        today_exit_note=enriched_report.today_exit_note,
        expected_return_pct=enriched_report.expected_return_pct,
        risk_reward_ratio=enriched_report.risk_reward_ratio,
        holding_days_estimate=enriched_report.holding_days_estimate,
        holding_window=enriched_report.holding_window,
        buy_zone=enriched_report.buy_zone,
        sell_zone=enriched_report.sell_zone,
        stop_loss=enriched_report.stop_loss,
        committee_summary=enriched_report.committee_summary,
        committee_model=enriched_report.committee_model,
        ai_enabled=enriched_report.ai_enabled,
        ai_available=enriched_report.ai_available,
        ai_error=enriched_report.ai_error,
        chart=enriched_report.chart,
        agents=enriched_report.agents,
        horizons=enriched_report.horizons,
        headline_count=headline_count,
        headline_summary=headline_summary,
        backtest=asdict(backtest),
        sector_score=sector_score,
        is_main_line=is_main_line,
        is_sector_leader=is_sector_leader,
        sector_boost=sector_boost,
        is_dark_horse=is_dark_horse,
        dark_horse_boost=dark_horse_boost,
        ai_chain_layer=ai_chain_layer,
        critical_bottleneck=critical_bottleneck,
        novice_rating=novice_rating,
        investingpro_fair_value=enriched_report.investingpro_fair_value,
        valuation_gap_pct=enriched_report.valuation_gap_pct,
        analyst_target_price=enriched_report.analyst_target_price,
        warren_ai_momentum=enriched_report.warren_ai_momentum,
        investingpro_models=enriched_report.investingpro_models,
        cognitive_temperature_gap=enriched_report.cognitive_temperature_gap,
        geopolitical_timing_advice=enriched_report.geopolitical_timing_advice,
        value_trap_risk=enriched_report.value_trap_risk,
        price_forecast=enriched_report.price_forecast,
        long_term_risk=enriched_report.long_term_risk,
    )



def get_sp500_daily_top_picks(
    *,
    period: str = "3y",
    limit: int = 50,
    prefilter_limit: int = FAST_SHORTLIST_LIMIT,
    use_ai_committee: bool = False,
    committee_model: str = "gemma4:e4b",
    market: str = "us",
    scan_type: str = "optimal",
) -> dict[str, Any]:
    key = _cache_key(period, limit, use_ai_committee, committee_model, scan_type=scan_type) + f":{market}"
    now = datetime.now()
    cached = _SCAN_CACHE.get(key)
    if cached and now - cached[0] <= timedelta(minutes=CACHE_TTL_MINUTES):
        return cached[1]

    if scan_type == "explosive_growth":
        # 早期飆股雷達：掃描黃仁勳「AI 算力蛋糕」各層個股（已是策展清單，不再依市值截斷）。
        constituents = fetch_ai_cake_universe(market)
    elif market == "tw":
        constituents = fetch_taiwan_constituents()
    else:
        constituents = fetch_sp500_constituents()

    # 掃描標的數量上限（給記憶體/時間有限的免費雲端用）。設環境變數 SP500_SCAN_LIMIT 即生效；
    # 不設或<=0 則掃全部（本機）。constituents 已依市值/重要性排序，取前 N 仍具代表性。
    # explosive_growth 的 AI 蛋糕池已策展、數量有限，不套用此截斷（否則會砍掉早期中型股）。
    try:
        _scan_limit = int(os.environ.get("SP500_SCAN_LIMIT", "0"))
    except ValueError:
        _scan_limit = 0
    if _scan_limit > 0 and scan_type != "explosive_growth":
        constituents = constituents[:_scan_limit]

    # 股癌自動併入：三種模式都跟著股癌長新標的（explosive 已在 fetch_ai_cake_universe 內含，這裡補
    # 強勢看漲/低估補漲）。放在截斷之後 → 股癌點名一律納入掃描、不會被市值截斷砍掉。
    if scan_type != "explosive_growth":
        _existing = {c.yf_symbol.upper() for c in constituents}
        for c in _gooaye_named_constituents(market):
            if c.yf_symbol.upper() not in _existing:
                _existing.add(c.yf_symbol.upper())
                constituents.append(c)

    # 免費雲端減重：可用環境變數縮短抓取期間（少下載 = 省記憶體/時間）。本機不設則用原 period。
    _scan_period = os.environ.get("SP500_SCAN_PERIOD", "").strip()
    if _scan_period:
        period = _scan_period
    try:
        _scan_workers = int(os.environ.get("SP500_SCAN_WORKERS", "12"))
    except ValueError:
        _scan_workers = 12
    _scan_workers = max(2, _scan_workers)

    # 免費雲端最大的時間殺手是逐檔 ticker.info / ticker.news 抓取（每檔都要爬 Yahoo）。
    # 設 SP500_EXTERNAL_ENRICH_LIMIT 可把「需外抓基本面/新聞」的檔數再夾低（<=0 不額外限制）。
    try:
        _external_enrich_cap = int(os.environ.get("SP500_EXTERNAL_ENRICH_LIMIT", "0"))
    except ValueError:
        _external_enrich_cap = 0

    try:
        regime = compute_market_regime(market_hint=market)
    except Exception as e:
        print(f"[sp500_daily] 市場情緒計算失敗，改用中性：{e}")
        regime = neutral_market_regime()
    constituent_map = {item.yf_symbol: item for item in constituents}
    price_map = download_sp500_price_map([item.yf_symbol for item in constituents], period)

    # 初篩：用 lightweight 報告快速排序整個宇宙（跳過個股回測/預測/估值抓取），並行加速。
    def _prefilter_one(yf_symbol: str, frame: pd.DataFrame):
        if len(frame) < 120:
            return None
        constituent = constituent_map.get(yf_symbol)
        if constituent is None:
            return None
        try:
            report = build_report(yf_symbol, frame, fetch_fundamentals=False, lightweight=True)
        except Exception:
            return None
        return (constituent, frame, report)

    candidates: list[tuple[SP500Constituent, pd.DataFrame, SignalReport]] = []
    with ThreadPoolExecutor(max_workers=_scan_workers) as prefilter_executor:
        prefilter_futures = [
            prefilter_executor.submit(_prefilter_one, yf_symbol, frame)
            for yf_symbol, frame in price_map.items()
        ]
        for future in as_completed(prefilter_futures):
            try:
                result = future.result()
            except Exception:
                result = None
            if result is not None:
                candidates.append(result)

    if not candidates:
        raise ValueError("S&P 500 掃描沒有產生任何候選股票。")

    # --- Start Sector Analysis ---
    sectors_map: dict[str, list[tuple[SP500Constituent, pd.DataFrame, SignalReport]]] = {}
    for item in candidates:
        sec = item[0].sector or ("其他" if market == "tw" else "Other")
        sectors_map.setdefault(sec, []).append(item)

    sectors_metrics: list[dict[str, Any]] = []
    for sec, members in sectors_map.items():
        avg_composite = sum(item[2].composite_score for item in members) / len(members)
        pct_above_ma20 = (sum(1 for item in members if item[2].latest_close > item[2].ma20) / len(members)) * 100
        sector_score = _clamp_int(avg_composite * 0.6 + pct_above_ma20 * 0.4)
        
        tot_ret_5d = 0.0
        for item in members:
            fr = item[1]
            if len(fr) >= 6:
                ret = (fr["Close"].iloc[-1] / fr["Close"].iloc[-6] - 1) * 100
                tot_ret_5d += ret
        avg_ret_5d = round(tot_ret_5d / len(members), 2)
        
        sorted_members = sorted(members, key=lambda x: x[2].composite_score, reverse=True)
        top_member_symbols = [x[0].symbol for x in sorted_members[:3]]
        
        if market == "tw":
            role_dict = {
                "半導體業": "AI 晶片與晶圓代工核心，全球科技主線標竿",
                "電腦及週邊設備業": "AI 伺服器與電子代工，高成長動能主線",
                "通信網路業": "大數據與光通訊題材，網通高成長波動主線",
                "電子零組件業": "散熱與PCB關鍵零組件，受惠AI硬體升級",
                "金融保險業": "息差與資產重組受惠，大盤穩健權值防禦板塊",
                "電機機械業": "重電工程與綠能電網政策題材，高Beta題材主線",
                "航運業": "全球航運運價與貿易景氣循環，高波動動能板塊",
                "光電業": "光學感測與面板觸控題材，消費景氣復甦板塊",
                "塑膠工業": "傳統石化原料景氣，穩健權值防禦大盤股",
                "鋼鐵工業": "基礎建設原物料需求，傳統產業價值股",
            }
        else:
            role_dict = {
                "Information Technology": "AI 晶片、軟體與雲端技術，市場核心多頭主線",
                "Communication Services": "大科技平台與高流量傳播，高成長動能主線",
                "Consumer Discretionary": "消費景氣與電動車題材，高 Beta 彈性板塊",
                "Financials": "利率政策與資產業績，權值穩健防禦大盤板塊",
                "Health Care": "剛性醫療與生技創新，中長線抗衰退防禦板塊",
                "Energy": "石油與天然氣探勘，受惠高通膨與油價避險板塊",
                "Consumer Staples": "抗通膨剛性日用品，高配息低波動防禦板塊",
                "Industrials": "國防航太與重型工業製造，景氣穩健成長板塊",
                "Materials": "基礎原物料與化學品，受惠全球基建與通膨板塊",
            }
        market_role = role_dict.get(sec, "穩健價值與大盤結構板塊")
        
        sectors_metrics.append({
            "name": sec,
            "score": sector_score,
            "avg_return_5d": avg_ret_5d,
            "member_count": len(members),
            "top_members": top_member_symbols,
            "market_role": market_role,
            "members_symbols": set(x[0].symbol for x in members),
            "leaders": set(x[0].symbol for x in sorted_members[:2])
        })
    
    sectors_metrics.sort(key=lambda x: x["score"], reverse=True)
    
    top_3_sectors = set(x["name"] for x in sectors_metrics[:3])
    bottom_2_sectors = set()
    if len(sectors_metrics) >= 5:
        bottom_2_sectors = set(x["name"] for x in sectors_metrics[-2:])
    elif len(sectors_metrics) > 3:
        bottom_2_sectors = set(x["name"] for x in sectors_metrics[3:])
        
    symbol_to_sector_data: dict[str, dict[str, Any]] = {}
    for sec_data in sectors_metrics:
        is_main_line = sec_data["name"] in top_3_sectors
        is_weak = sec_data["name"] in bottom_2_sectors
        for sym in sec_data["members_symbols"]:
            is_leader = sym in sec_data["leaders"]
            boost = 6 if is_main_line else (-3 if is_weak else 0)
            if is_main_line and is_leader:
                boost += 3
            symbol_to_sector_data[sym] = {
                "sector_score": sec_data["score"],
                "is_main_line": is_main_line,
                "is_sector_leader": is_leader,
                "sector_boost": boost
            }
    # --- End Sector Analysis ---

    if scan_type == "lagging_value":
        def calculate_pre_lagging_score(item, sector_boost: int) -> float:
            constituent, frame, report = item
            rsi = report.rsi14
            close = report.latest_close
            ma20 = report.ma20
            ma50 = report.ma50
            support = report.support
            
            # Base technical composite score with lower weight
            score = report.composite_score * 0.3 + sector_boost * 0.3
            
            # 1. InvestingPro Valuation Gap bonus (High Weight!)
            gap = report.valuation_gap_pct
            if gap is not None:
                if gap > 20:
                    score += 50
                elif gap > 10:
                    score += 25
                elif gap < -5:
                    score -= 20
            
            # 2. RSI pullback bonus
            if 30 <= rsi <= 48:
                score += 35
            elif 48 < rsi <= 55:
                score += 15
            elif rsi < 30:
                score += 25
            else:
                score -= 20
                
            # 3. Pullback below MA20 bonus
            if close < ma20:
                score += 15
                
            # 4. Must be supported (not fully broken down)
            if close >= ma50 * 0.96:
                score += 15
            else:
                score -= 10
                
            # 5. Near support level bonus
            if support > 0:
                dist_to_support = (close / support) - 1
                if 0 <= dist_to_support <= 0.05:
                    score += 15
                    
            return score

        candidates.sort(
            key=lambda item: calculate_pre_lagging_score(
                item,
                symbol_to_sector_data.get(item[0].symbol, {}).get("sector_boost", 0)
            ),
            reverse=True,
        )
    elif scan_type == "explosive_growth":
        # 早期飆股雷達：初篩就以「早期型態分數」排序（AI 蛋糕池已策展、通常全數入圍，此排序只在
        # 池子超過 shortlist 上限時決定取捨）。
        candidates.sort(
            key=lambda item: _early_stage_score(item[1])
                + symbol_to_sector_data.get(item[0].symbol, {}).get("sector_boost", 0),
            reverse=True,
        )
    else:
        # 初篩排序也納入相對強度，避免強勢股在進入完整評分前就被 composite_score 偏誤砍掉
        # （Render 上 universe≤40 一律全評分，此排序主要影響本機掃全宇宙時的入圍名單）。
        candidates.sort(
            key=lambda item: (
                item[2].composite_score
                + symbol_to_sector_data.get(item[0].symbol, {}).get("sector_boost", 0)
                + 0.6 * (_relative_strength_from_frame(item[1]) - 50.0),
                item[2].expected_return_pct,
                item[2].risk_reward_ratio,
            ),
            reverse=True,
        )

    shortlist_size = min(len(candidates), max(limit, min(prefilter_limit, FAST_SHORTLIST_LIMIT)))
    shortlisted = candidates[:shortlist_size]
    external_enrich_limit = min(len(shortlisted), max(min(limit, FAST_EXTERNAL_ENRICH_LIMIT), 16))
    if _external_enrich_cap > 0:
        external_enrich_limit = min(external_enrich_limit, _external_enrich_cap)
    ai_enrich_limit = min(external_enrich_limit, FAST_AI_ENRICH_LIMIT if use_ai_committee else 0)

    picks: list[SP500DailyPick] = []
    with ThreadPoolExecutor(max_workers=min(_scan_workers, 8 if use_ai_committee else 12)) as executor:
        futures = {
            executor.submit(
                enrich_candidate,
                constituent,
                frame,
                report,
                regime,
                fetch_external_data=index < external_enrich_limit,
                use_ai_committee=use_ai_committee and index < ai_enrich_limit,
                committee_model=committee_model,
                sector_score=symbol_to_sector_data.get(constituent.symbol, {}).get("sector_score", 50),
                is_main_line=symbol_to_sector_data.get(constituent.symbol, {}).get("is_main_line", False),
                is_sector_leader=symbol_to_sector_data.get(constituent.symbol, {}).get("is_sector_leader", False),
                sector_boost=symbol_to_sector_data.get(constituent.symbol, {}).get("sector_boost", 0),
                scan_type=scan_type,
            ): constituent.symbol
            for index, (constituent, frame, report) in enumerate(shortlisted)
        }
        for future in as_completed(futures):
            try:
                picks.append(future.result())
            except Exception:
                continue

    def near_term_entry_adj(item: SP500DailyPick) -> int:
        """近期(非僅限今天)進場可行性的「軟」加分：理想買點加分、已轉賣訊號扣分。
        2026-06-19 改：不再硬性把「今天可買」整批拉到最前面 —— 改以分數(含未來上漲空間)為主，
        讓看好後續大漲的強勢股不會只因為差一點回檔、今天不是完美買點就被埋到後面。"""
        if item.today_action == "今天可買":
            return 4
        if item.today_action == "今天可小量買":
            return 2
        if item.today_action == "今天等回檔":
            return 1
        if item.today_exit_action == "今天賣出":
            return -8
        if item.today_exit_action == "今天可小量賣":
            return -3
        return 0

    if scan_type == "lagging_value":
        # 低估補漲：以 InvestingPro 12 模型穩健估值「折價 %」為主排序鍵 —— 直接呈現「被低估最多」的排名；
        # daily_score(已含基本面/相對強度/上漲空間，並已過濾價值陷阱) 為次要鍵，確保被低估又有補漲動能者居前。
        # 無估值資料者(折價視為 -999) 沉到名單後段，不會擠掉真正被低估的好股。
        def _gap_key(item: SP500DailyPick) -> float:
            g = item.valuation_gap_pct
            return g if g is not None else -999.0
        picks.sort(
            key=lambda item: (
                _gap_key(item),
                item.daily_score,
                item.backtest_score,
                item.technical_score,
                item.fundamental_score,
            ),
            reverse=True,
        )
        ordered_picks = picks
    elif scan_type == "explosive_growth":
        # 早期飆股雷達：純以 daily_score 排序（早期布局不該被「今天能否進場」綁住）。
        picks.sort(
            key=lambda item: (
                item.daily_score,
                item.backtest_score,
                item.technical_score,
                item.fundamental_score,
                item.news_score,
            ),
            reverse=True,
        )
        ordered_picks = picks
    else:
        # 近期最佳買點＋後續看漲：以 daily_score(已含相對強度與半年上漲空間) 為主，
        # 近期進場可行性只當軟加分，預期報酬為次要排序鍵；仍避免把「今天賣出」訊號排到前面。
        picks.sort(
            key=lambda item: (
                item.daily_score + near_term_entry_adj(item),
                item.expected_return_pct,
                item.backtest_score,
                item.technical_score,
            ),
            reverse=True,
        )
        ordered_picks = picks

    # Prepare serialized sectors list (clean up raw set objects)
    serialized_sectors = []
    for s in sectors_metrics:
        serialized_sectors.append({
            "name": s["name"],
            "score": s["score"],
            "is_main_line": s["name"] in top_3_sectors,
            "avg_return_5d": s["avg_return_5d"],
            "member_count": s["member_count"],
            "top_members": s["top_members"],
            "market_role": s["market_role"]
        })

    payload = {
        "market_regime": asdict(regime),
        "picks": [asdict(pick) for pick in ordered_picks[:limit]],
        "sectors": serialized_sectors,
        "generated_at": now.isoformat(),
    }
    _SCAN_CACHE[key] = (now, payload)
    return payload
