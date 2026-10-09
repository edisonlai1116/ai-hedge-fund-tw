"""單一策略：AI 科技股「Sharpe 動能」排名輪動。2026-10-06 全面改版（同日依使用者偏好改為 AI 科技股票池）。

為什麼換掉舊框架：舊的「綜合分數」是一堆手調規則（RSI 加減分、回檔加分、大師 agent 投票…），
自己的檢討紀錄就寫著與未來報酬的相關性近乎 0。這次改成先做因子研究、只留回測驗證過的訊號：
  因子研究（S&P 500 + 台股 570 檔、2017~2026、前後兩段樣本外）：6 個月風險調整動能
  （日報酬平均 / 標準差）在台股最強且穩定，美股多數價格因子弱且正負翻轉、此因子最穩定。

股票池（只排科技／AI 相關）：
  - 美股：S&P 500 資訊科技類股 + GOOGL/META/NFLX/AMZN/TSLA + AI 電力/基建（CEG、VST、GEV、ETN、VRT…）
    + 非 S&P 的 AI 股（TSM、ASML、ARM、ALAB、CRDO…）。不含電信、傳媒、金融、醫療。
  - 台股：半導體、AI 伺服器、散熱、PCB/CCL、記憶體、IC 設計約 55 檔。

投組回測（2017-07 ~ 2026-10、月調、每次換股成本 0.2%）：
  - 美股 持有前 10、跌出前 50 才賣：年化 46.4%、最大回撤 -38%；
    同池等權持有 34.5%、QQQ 21.2%、VOO 15.3%。前後兩段（40.3% / 53.3%）都勝過同池等權持有。
  - 台股 持有前 20、跌出前 30 才賣：年化 45.6%、最大回撤 -38%；
    同池等權持有 41.1%（回撤 -34%）、0050 24.7%。選股優勢比美股小，主要賺的是族群本身。
  注意：「AI 科技股」股票池本身是用現在眼光挑的贏家，有明顯倖存者/後見之明偏差，
  實際報酬會比回測低很多，也可能在 AI 族群轉弱時大幅落後大盤。

規則（網站、每日報告、自動提醒、回測全部共用這一份）：
  - 分數 = 近 126 個交易日日報酬的平均 / 標準差（年化顯示）。
  - 持有排名前 TOP_N[市場] 檔、等權；每月第一個交易日檢查。
  - 持股跌出前 KEEP_N[市場] 名才賣出換股。
  - ETF（台股 00 開頭、常見美股 ETF）屬核心部位，不套用個股輪動。
"""
from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

LOOKBACK = 126
TOP_N = {"us": 10, "tw": 20}
KEEP_N = {"us": 50, "tw": 30}
UNDERWEIGHT_RATIO = 0.7   # 部位 < 目標 × 0.7 → 加碼

# 長線低檔布局（2026-10-07 依使用者風格新增，point-in-time 回測驗證）：
#   長線贏家（3 年報酬 > 0）自 52 週高點回落 ≥ 30% → 買進、持有 12 個月。
#   美股 2017-2026：年化 44.3%、勝率 81%、Profit Factor 17.9（樣本內 40.2% / 樣本外 48.2%）；
#   台股：39.5%（40.5% / 39.0%）。等待「止穩」或設 -35% 停損都會降低報酬（賣在低點），故不採用。
#   CEG 2025-03-21、LITE 2026-07-30、AVGO 2025-04-09、VST 2025-03-14 皆會觸發。
LOWENTRY_DD = -0.30
LOWENTRY_WATCH_DD = -0.20
LOWENTRY_LT_YEARS = 3
LOWENTRY_HOLD_MONTHS = 12
LOWENTRY_SLOTS = 10
ALLOCATION = {"lowentry": 0.70, "momentum": 0.30}   # 使用者選擇：低檔布局為主
CONCENTRATION_PCT = 20.0  # 單檔 > 總資產 20% → 減碼（集中度風險）

# 2026-10-07 市場情緒研究（point-in-time，T-1 收盤訊號 → T 開盤成交，成本 0.2%；自建恐懼貪婪見 src.strategy.sentiment）：
#   採用 1）低檔區「沒用到的槽位」資金放動能名單，而不是現金：美股 46.1% → 50.7%、台股 42.1% → 45.1%（樣本內外皆改善）。
#   採用 2）恐懼貪婪 < 25（極度恐懼）時，月調「只買不賣」——不在恐慌低點砍掉動能股：美股 → 52.2%（回撤 -39.9% → -38.6%）、
#          台股 → 45.8%；門檻 15~45 全部 ≥ 不用，樣本內外皆改善。
#   不採用（回測變差或兩市場不一致）：恐慌時放寬低檔門檻到 -20/-25%、恐慌時加開低檔槽位、貪婪時暫停買進、
#          只在恐懼時才做低檔、新資金等恐懼才投入（比立即投入少 0.4~1.8%）、每日檢查動能出場。
IDLE_TO_MOMENTUM = True
PANIC_NO_SELL_FG = 25.0
# 動能排名混入 virattt/ai-hedge-fund 技術分析師分數（趨勢/動能/均值回歸加權，src.agents.technicals）：
#   排名分數 = (1-w) × 動能百分位 + w × 技術分數百分位。完整策略（低檔 + 情緒規則）回測、取報酬最高者：
#   台股 w=0.7：45.8% → 47.5%（樣本內 44.7→46.6、樣本外 47.1→48.5，w=0.3/0.5 也都較好）；
#   美股 w=0 最好（52.2%；w=0.3/0.5/0.7 為 51.6/50.0/47.8%、回撤變大）→ 美股維持純動能。
TECH_BLEND = {"us": 0.0, "tw": 0.7}
# 不追大長紅（2026-10-07 依使用者偏好「買在漲之前、不買剛噴完一根大的」）：
#   近 SPIKE_DAYS 日任一日收盤漲幅 ≥ SPIKE_PCT → 不追。
#   低檔布局：掛「大漲前一日收盤價」限價，LIMIT_VALID_DAYS 個交易日內沒回到就放棄（20 日內不再追）。
#   動能月調：等大長紅超過 SPIKE_DAYS 日、且仍在前 N 名才買。
#   回測（完整策略）：門檻 10% 美股 52.2→52.3%、台股 47.5→47.5%（不傷報酬）；門檻 7% 台股少 1.5~3%，故用 10%。
#   「看到反彈就跳過」反而少賺 2~5%（反彈中的低檔股之後常續漲），故用限價等拉回而非跳過。
SPIKE_PCT = 0.10
SPIKE_DAYS = 3
LIMIT_VALID_DAYS = 20
LOWENTRY_REENTRY_DAYS = 60   # 離開低檔區不到 60 個交易日又跌回 → 不算「新訊號」
TECH_BARS = 300
TPE = timezone(timedelta(hours=8))

US_ETFS = {
    "VOO", "SPY", "IVV", "VTI", "QQQ", "QQQM", "DIA", "IWM", "VT", "VXUS", "SCHD", "VIG", "VUG",
    "SOXX", "SMH", "XLK", "TLT", "IEF", "SHY", "BND", "AGG", "GLD", "IAU", "SGOV", "BIL", "TQQQ", "SOXL",
}
# 美股 AI 科技池：S&P 500「資訊科技」類股（每日由成分股清單動態取得）+ 下列 AI 相關個股。
US_AI_EXTRAS = [
    "GOOGL", "META", "NFLX", "RDDT", "AMZN", "TSLA",                  # 平台／AI 應用
    "CEG", "VST", "TLN", "NRG", "GEV", "ETN", "VRT", "PWR",           # AI 電力與資料中心基建
    "TSM", "ASML", "ARM", "ALAB", "CRDO", "NBIS", "CRWV", "OKLO", "BE",
    "SNOW", "NET", "MDB", "SHOP", "UBER",
]
# 成分股清單抓不到時的備援（2026-10 S&P 500 資訊科技類股快照）。
US_IT_SNAPSHOT = (
    "ACN ADBE AMD AKAM APH ADI AAPL AMAT APP ANET ADSK AVGO CDNS CDW CIEN CSCO CTSH COHR GLW CRWD DDOG DELL "
    "FFIV FICO FSLR FLEX FTNT IT GEN GDDY HPE HPQ IBM INTC INTU JBL KEYS KLAC LRCX LITE MRVL MCHP MU MSFT "
    "MPWR MSI NTAP NVDA NXPI ON ORCL PLTR PANW PTC QCOM ROP CRM SNDK STX NOW SWKS SMCI SNPS TEL TDY TER TXN "
    "TRMB TYL VRSN WDAY ZBRA WDC"
).split()
TW_AI_UNIVERSE = [
    "2330", "2454", "2317", "2382", "2308", "3711", "2303", "3034", "3037", "3231", "2376", "2377",
    "2357", "2395", "3008", "2474", "2345", "3661", "4938", "2379", "3017", "2327", "8299.TWO", "2408",
    "6669", "2301", "2421", "6409", "3443", "2449", "3533", "2368", "3044", "6274.TWO", "2383", "3653",
    "3324.TWO", "2059", "6415", "5274.TWO", "3529.TWO", "6488.TWO", "3105.TWO", "2344", "2337", "6531",
    "3036", "2360", "2404", "6139", "4966", "3406", "5269", "2356", "2353", "2324",
]
# 台股族群（類股情緒用；股票池內等權指數）
TW_GROUPS = {
    "晶圓代工/矽晶圓": ["2330.TW", "2303.TW", "6488.TWO", "3105.TWO"],
    "IC 設計": ["2454.TW", "3034.TW", "2379.TW", "3661.TW", "3443.TW", "5274.TWO", "3529.TWO", "6415.TW", "5269.TW", "6531.TW"],
    "AI 伺服器/ODM": ["2317.TW", "2382.TW", "3231.TW", "2376.TW", "2377.TW", "2357.TW", "6669.TW", "2356.TW", "4938.TW", "2324.TW", "2353.TW"],
    "散熱/機構": ["3017.TW", "3324.TWO", "3653.TW", "2059.TW", "2421.TW"],
    "PCB/CCL/連接": ["2383.TW", "6274.TWO", "2368.TW", "3044.TW", "3037.TW", "3533.TW", "2345.TW"],
    "記憶體": ["2408.TW", "2344.TW", "2337.TW", "8299.TWO"],
    "封測/設備": ["3711.TW", "2449.TW", "2360.TW", "2404.TW", "6139.TW"],
    "電源/零組件": ["2308.TW", "6409.TW", "2301.TW", "2327.TW", "3008.TW", "3406.TW", "2474.TW", "2395.TW", "3036.TW"],
}
TW_AI_NAMES = {
    "2449": "京元電子", "3533": "嘉澤", "2368": "金像電", "3044": "健鼎", "6274": "台燿", "2383": "台光電",
    "3653": "健策", "3324": "雙鴻", "2059": "川湖", "6415": "矽力-KY", "5274": "信驊", "3529": "力旺",
    "6488": "環球晶", "3105": "穩懋", "2344": "華邦電", "2337": "旺宏", "6531": "愛普", "3036": "文曄",
    "2360": "致茂", "2404": "漢唐", "6139": "亞翔", "4966": "譜瑞-KY", "3406": "玉晶光", "5269": "祥碩",
    "2356": "英業達", "2353": "宏碁", "2324": "仁寶", "3443": "創意",
}


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------
def is_etf(symbol: str) -> bool:
    s = symbol.upper()
    if s.endswith((".TW", ".TWO")):
        return s.split(".")[0].startswith("00")
    return s in US_ETFS


def market_of(symbol: str) -> str:
    return "tw" if symbol.upper().endswith((".TW", ".TWO")) else "us"


def to_yf(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." in t:
        return t
    return f"{t}.TW" if t[:1].isdigit() else t


def sp500_it_symbols() -> List[str]:
    """S&P 500 資訊科技類股（動態）；失敗回快照。"""
    try:
        import requests
        csv = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            timeout=30,
        ).text
        df = pd.read_csv(io.StringIO(csv))
        syms = [str(s).replace(".", "-") for s in df[df["GICS Sector"] == "Information Technology"]["Symbol"]]
        if len(syms) > 50:
            return syms
    except Exception as e:
        print(f"[strategy] S&P 500 清單下載失敗，改用快照：{e}")
    return list(US_IT_SNAPSHOT)


def universe(market: str) -> List[str]:
    if market == "us":
        syms = sp500_it_symbols() + US_AI_EXTRAS
    else:
        syms = [to_yf(t) for t in TW_AI_UNIVERSE]
    return [s for s in dict.fromkeys(syms) if not is_etf(s)]


# ---------------------------------------------------------------------------
# 分數
# ---------------------------------------------------------------------------
def sharpe_momentum(close: pd.Series, lookback: int = LOOKBACK) -> Optional[float]:
    """近 lookback 日的日報酬平均 / 標準差（年化）。資料不足回 None。"""
    c = close.dropna()
    if len(c) < lookback + 1:
        return None
    r = c.pct_change().iloc[-lookback:]
    sd = float(r.std())
    if not sd or math.isnan(sd):
        return None
    return float(r.mean()) / sd * math.sqrt(252)


def download_closes(symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """批次下載 OHLCV（auto_adjust 還原權息）。回傳 {symbol: DataFrame}。"""
    import yfinance as yf
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        data = yf.download(chunk, period=period, auto_adjust=True, progress=False, threads=True, group_by="ticker")
        if data is None or data.empty:
            continue
        for s in chunk:
            try:
                f = data[s] if isinstance(data.columns, pd.MultiIndex) else data
                f = f.dropna(subset=["Close"])
                if len(f) > 30:
                    out[s] = f
            except Exception:
                continue
    return out


def _ignition(f: pd.DataFrame) -> Optional[Dict]:
    """近 3 日爆量長紅點火（與 simple_signal.detect_ignition 同定義），供事件提醒用。"""
    try:
        from src.simple_signal import detect_ignition
        ign = detect_ignition(f)
        return ign if ign.get("ignition_days_ago") is not None else None
    except Exception:
        return None


def tech_detail(f: pd.DataFrame) -> Optional[Dict]:
    """virattt/ai-hedge-fund 技術分析師（src.agents.technicals）：綜合分數 0~100、訊號、各子策略分數。只用最後 TECH_BARS 根。"""
    try:
        from src.agents import technicals as T
        df = f.tail(TECH_BARS).rename(columns=str.lower)
        if len(df) < 260:
            return None
        comps = {"trend": T.calculate_trend_signals(df), "mean_reversion": T.calculate_mean_reversion_signals(df),
                 "momentum": T.calculate_momentum_signals(df), "volatility": T.calculate_volatility_signals(df),
                 "stat_arb": T.calculate_stat_arb_signals(df)}
        c = T.weighted_signal_combination(comps, T.TECHNICAL_WEIGHTS)
        v = c.get("score")
        if v is None or not np.isfinite(v):
            return None
        sub = lambda k: (round(float(comps[k]["score"]), 0) if comps[k].get("score") is not None else None)
        return {"score": round(float(v), 1), "signal": c.get("signal"), "confidence": round(float(c.get("confidence") or 0) * 100),
                "trend": sub("trend"), "momentum": sub("momentum"), "mean_reversion": sub("mean_reversion"),
                "volatility": sub("volatility"), "stat_arb": sub("stat_arb")}
    except Exception:
        return None


def tech_score(f: pd.DataFrame) -> Optional[float]:
    """virattt 技術分析師綜合分數（0~100）；資料不足回 None。"""
    d = tech_detail(f)
    return d["score"] if d else None


def blend_rank_scores(mom: pd.Series, tech: Optional[pd.Series], w: float) -> pd.Series:
    """排名分數 = (1-w) × 動能百分位 + w × 技術百分位（技術缺值視為 0.5）。w=0 時就是純動能百分位。"""
    a = mom.dropna().rank(pct=True)
    if not w or tech is None:
        return a
    return (1 - w) * a + w * tech.rank(pct=True).reindex(a.index).fillna(0.5)


def score_row(symbol: str, f: pd.DataFrame) -> Optional[Dict]:
    sc = sharpe_momentum(f["Close"])
    if sc is None:
        return None
    c = f["Close"].dropna()
    ma200 = float(c.tail(200).mean()) if len(c) >= 200 else None
    row = {
        "symbol": symbol,
        "score": round(sc, 3),
        "close": round(float(c.iloc[-1]), 2),
        "ret_6m_pct": round((float(c.iloc[-1]) / float(c.iloc[-LOOKBACK - 1]) - 1) * 100, 1),
        "ret_1m_pct": round((float(c.iloc[-1]) / float(c.iloc[-22]) - 1) * 100, 1) if len(c) > 22 else None,
        "vol_ann_pct": round(float(c.pct_change().iloc[-LOOKBACK:].std()) * math.sqrt(252) * 100, 1),
        "above_ma200": (float(c.iloc[-1]) > ma200) if ma200 else None,
        "day_change_pct": round((float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100, 2),
    }
    hi252 = float(c.tail(252).max())
    dd = float(c.iloc[-1]) / hi252 - 1 if hi252 > 0 else None
    n3 = 252 * LOWENTRY_LT_YEARS
    r3 = float(c.iloc[-1] / c.iloc[-n3 - 1] - 1) if len(c) > n3 else None
    row["dd_52w_pct"] = round(dd * 100, 1) if dd is not None else None
    row["ret_3y_pct"] = round(r3 * 100, 1) if r3 is not None else None
    row["high_52w"] = round(hi252, 2)
    lt_winner = r3 is not None and r3 > 0
    row["low_entry"] = bool(lt_winner and dd is not None and dd <= LOWENTRY_DD)
    row["low_entry_watch"] = bool(lt_winner and dd is not None and LOWENTRY_DD < dd <= LOWENTRY_WATCH_DD)
    row["low_entry_price"] = round(hi252 * (1 + LOWENTRY_DD), 2) if lt_winner else None
    row["long_term_broken"] = r3 is not None and r3 <= 0
    # 連續幾天處於低檔區；若這段之前 60 日內也在低檔區（只是短暫反彈出去又跌回）→ 不算新訊號
    if row["low_entry"]:
        hi = c.rolling(252, min_periods=200).max()
        lt_ok = (c / c.shift(n3) - 1) > 0
        ok = ((c / hi - 1) <= LOWENTRY_DD) & lt_ok
        tail = ok.iloc[::-1].tolist()
        streak = next((i for i, v in enumerate(tail) if not v), len(tail))
        row["low_entry_days"] = streak
        row["low_entry_reentry"] = any(tail[streak:streak + LOWENTRY_REENTRY_DAYS])
    # 不追大長紅 / 是否已從低點反彈（買在漲之前）
    rets = c.pct_change()
    last = rets.tail(SPIKE_DAYS)
    big = last[last >= SPIKE_PCT]
    if len(big):
        bday = big.index[0]
        pos_b = c.index.get_loc(bday)
        row["spike"] = {"date": pd.Timestamp(bday).strftime("%Y-%m-%d"), "gain_pct": round(float(big.iloc[0]) * 100, 1),
                        "limit_price": round(float(c.iloc[pos_b - 1]), 2) if pos_b >= 1 else None}
    row["bounce_20d_pct"] = round((float(c.iloc[-1]) / float(c.tail(20).min()) - 1) * 100, 1) if len(c) >= 20 else None
    vd = tech_detail(f)   # 每檔都算（網站顯示 ai-hedge-fund 技術分析師）；只有 TECH_BLEND > 0 的市場混入排名
    row["virattt"] = vd
    row["tech_score"] = vd["score"] if vd else None
    ign = _ignition(f)
    if ign:
        row["ignition"] = {k: ign.get(k) for k in ("ignition_days_ago", "ignition_gain_pct", "ignition_volume_ratio", "ignition_low")}
    return row


def no_chase_note(row: Optional[Dict]) -> str:
    """近 3 日有大長紅 → 不追，掛大漲前收盤價。"""
    sp = (row or {}).get("spike")
    if not sp:
        return ""
    lim = f"掛 {sp['limit_price']} 限價等拉回，{LIMIT_VALID_DAYS} 個交易日內沒回到就放棄" if sp.get("limit_price") else "等拉回再買"
    return f" ⚠️ {sp['date']} 單日大漲 +{sp['gain_pct']}%，不追：{lim}（低檔布局）；動能名單則等大漲超過 {SPIKE_DAYS} 天再買。"


def _pct(v, signed: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:+.0f}%" if signed else f"{v:.0f}%"


def zone_of(rank: int, market: str) -> str:
    return "buy" if rank <= TOP_N[market] else ("hold" if rank <= KEEP_N[market] else "out")


def rank_market(market: str, price_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
    syms = universe(market)
    price_map = price_map if price_map is not None else download_closes(syms, period="4y")
    rows = [r for s in syms if s in price_map for r in [score_row(s, price_map[s])] if r]
    w = TECH_BLEND.get(market, 0.0)
    rs = blend_rank_scores(pd.Series({r["symbol"]: r["score"] for r in rows}),
                           pd.Series({r["symbol"]: r.get("tech_score") for r in rows}, dtype=float) if w else None, w)
    for r in rows:
        r["rank_score"] = round(float(rs.get(r["symbol"], 0.0)), 4)
    rows.sort(key=lambda r: r["rank_score"], reverse=True)
    from src.ranking.scoring import price_location, quality_tier
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["zone"] = zone_of(i, market)
        r["quality_tier"] = quality_tier(i, len(rows))
        r["price_location"] = price_location(r["dd_52w_pct"] / 100 if r.get("dd_52w_pct") is not None else None)
    names = _names(market)
    for r in rows:
        r["name"] = names.get(r["symbol"].split(".")[0], "")
    low = sorted([r for r in rows if r.get("low_entry")], key=lambda r: r["dd_52w_pct"])
    watch = sorted([r for r in rows if r.get("low_entry_watch")], key=lambda r: r["dd_52w_pct"])
    return {
        "universe_size": len(rows),
        "low_entry": [r["symbol"] for r in low],
        "low_entry_watch": [r["symbol"] for r in watch],
        "top_n": TOP_N[market],
        "tech_blend": w,
        "threshold_top": rows[TOP_N[market] - 1]["score"] if len(rows) >= TOP_N[market] else None,
        "keep_n": KEEP_N[market],
        "threshold_keep": rows[KEEP_N[market] - 1]["score"] if len(rows) >= KEEP_N[market] else None,
        "rows": rows,
    }


def _names(market: str) -> Dict[str, str]:
    if market != "tw":
        return {}
    try:
        from src.pipeline.daily_report import TW_NAMES
        return {**dict(TW_NAMES), **TW_AI_NAMES}
    except Exception:
        return {}


def next_rebalance(today: Optional[date] = None) -> str:
    """下次檢查日：下個月第一個平日。"""
    today = today or datetime.now(TPE).date()
    first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    while first.weekday() >= 5:
        first += timedelta(days=1)
    return first.isoformat()


def is_rebalance_window(today: Optional[date] = None) -> bool:
    """每月前 3 個平日內視為調整窗口（避開假日／排程漏跑）。"""
    today = today or datetime.now(TPE).date()
    d, n = today.replace(day=1), 0
    while d <= today:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return today.weekday() < 5 and n <= 3


def us_groups(symbols: Iterable[str]) -> Dict[str, List[str]]:
    """美股股票池依主題分組（src.ranking.themes）；不在主題表的歸到「其他科技」。"""
    try:
        from src.ranking.themes import theme_of
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for s in symbols:
        t = theme_of(s)
        out.setdefault(t.theme if t else "其他科技", []).append(s)
    return {k: v for k, v in out.items() if len(v) >= 2}


def momentum_share(n_low_entry: int) -> float:
    """動能實際資金占比：低檔區沒用到的槽位資金也放動能（IDLE_TO_MOMENTUM，回測驗證）。"""
    if not IDLE_TO_MOMENTUM:
        return ALLOCATION["momentum"]
    used = min(n_low_entry, LOWENTRY_SLOTS) / LOWENTRY_SLOTS
    return ALLOCATION["momentum"] + ALLOCATION["lowentry"] * (1 - used)


def panic_mode(report: Dict) -> bool:
    fg = ((report.get("sentiment") or {}).get("fear_greed") or {}).get("score")
    return fg is not None and fg < PANIC_NO_SELL_FG


def build_strategy_report() -> Dict:
    now = datetime.now(TPE)
    markets = {}
    prices: Dict[str, Dict[str, pd.DataFrame]] = {}
    for m in ("us", "tw"):
        try:
            prices[m] = download_closes(universe(m), period="4y")
            markets[m] = rank_market(m, prices[m])
            markets[m]["momentum_share"] = round(momentum_share(len(markets[m]["low_entry"])), 3)
        except Exception as e:
            print(f"[strategy] {m} 排名失敗：{e}")
    sentiment = None
    try:
        from src.strategy.sentiment import market_sentiment
        groups, uclose = {}, {}
        if "us" in prices:
            groups.update({f"美股·{k}": v for k, v in us_groups(prices["us"]).items()})
            uclose.update({s: f["Close"] for s, f in prices["us"].items()})
        if "tw" in prices:
            groups.update({f"台股·{k}": v for k, v in TW_GROUPS.items()})
            uclose.update({s: f["Close"] for s, f in prices["tw"].items()})
        ucl = pd.DataFrame(uclose) if uclose else None
        if ucl is not None:
            ucl.index = pd.to_datetime(ucl.index)
            if ucl.index.tz is not None:
                ucl.index = ucl.index.tz_localize(None)
            ucl = ucl.groupby(ucl.index.normalize()).last()   # 台美股混合時區 → 以日期對齊
        sentiment = market_sentiment(groups, ucl)
    except Exception as e:
        print(f"[strategy] 市場情緒計算失敗（不影響排名）：{type(e).__name__}: {e}")
    return {
        "sentiment": sentiment,
        "generated_at": now.isoformat(timespec="seconds"),
        "strategy": {
            "name": "長線低檔布局 70% ＋ 動能輪動 30%",
            "lookback_days": LOOKBACK, "top_n": TOP_N, "keep_n": KEEP_N, "allocation": ALLOCATION,
            "low_entry_rule": (f"長線贏家（{LOWENTRY_LT_YEARS} 年報酬 > 0）自 52 週高點回落 ≥ {abs(LOWENTRY_DD):.0%} 只代表「價格跌深」；"
                               "是否買進看排名 Tier（A 可買、B 分批、C 觀察、D 不因跌深而買）與回撤分類，"
                               f"持有 {LOWENTRY_HOLD_MONTHS} 個月；回落 {abs(LOWENTRY_WATCH_DD):.0%}~{abs(LOWENTRY_DD):.0%} 列入觀察。"),
            "rule": (f"資金 {ALLOCATION['lowentry']:.0%} 給長線低檔布局、{ALLOCATION['momentum']:.0%} 給動能輪動"
                     f"（美股前 {TOP_N['us']} 名、跌出前 {KEEP_N['us']} 名才賣；台股前 {TOP_N['tw']} 名、跌出前 {KEEP_N['tw']} 名才賣）。"
                     f"持股只有在動能也轉弱且 {LOWENTRY_LT_YEARS} 年長線趨勢破壞時才建議換股。"
                     f"低檔區沒用到的槽位資金放動能名單（不留現金）；恐懼貪婪 < {PANIC_NO_SELL_FG:.0f}（極度恐懼）時月調只買不賣。"),
            "idle_to_momentum": IDLE_TO_MOMENTUM, "panic_no_sell_fg": PANIC_NO_SELL_FG, "tech_blend": TECH_BLEND,
            "ranking_rule": (f"美股排名＝Sharpe 動能；台股排名＝動能 {1 - TECH_BLEND['tw']:.0%} ＋ virattt 技術分析師分數 "
                             f"{TECH_BLEND['tw']:.0%}（皆為股票池內百分位）。"),
            "next_rebalance": next_rebalance(now.date()),
            "in_rebalance_window": is_rebalance_window(now.date()),
        },
        "markets": markets,
        "disclaimer": "規則化訊號，回測有倖存者偏差，非投資建議。",
    }


# ---------------------------------------------------------------------------
# 持股評估（網站「我的持股」與自動提醒共用）
# ---------------------------------------------------------------------------
def _rank_for_score(row: Dict, rows: List[Dict]) -> int:
    """非股票池個股：以同樣的排名分數（動能 / 技術百分位）插入股票池，得到等效名次。"""
    if not rows:
        return 1
    n = len(rows)
    w = TECH_BLEND.get(market_of(row["symbol"]), 0.0)
    pm = sum(1 for r in rows if r["score"] <= row["score"]) / n
    rs = pm
    if w:
        t = row.get("tech_score")
        pt = 0.5 if t is None else sum(1 for r in rows if r.get("tech_score") is not None and r["tech_score"] <= t) / n
        rs = (1 - w) * pm + w * pt
    row["rank_score"] = round(rs, 4)
    return 1 + sum(1 for r in rows if r.get("rank_score", 0) > rs)


def low_view(row: Optional[Dict], report: Dict, universe_size: Optional[int] = None) -> Optional[Dict]:
    """跌深長線贏家的「低檔狀態 + 新資金建議」（與機會評分同一函式 scoring.low_price_recommendation）。
    Price Location 只說明跌多深；是否值得買由排名 Tier、回撤分類、獨立 thesis 證據決定。不是跌深 → None。"""
    if not row or not row.get("low_entry"):
        return None
    from src.ranking.scoring import low_price_recommendation, price_location, quality_tier
    info = (report.get("drawdown_types") or {}).get(row["symbol"]) or {}
    n = universe_size or (report.get("markets", {}).get(market_of(row["symbol"]), {}) or {}).get("universe_size")
    tier = row.get("quality_tier") or (quality_tier(row.get("rank"), n) if row.get("rank") and n else None)
    loc = row.get("price_location") or price_location((row.get("dd_52w_pct") or 0) / 100)
    return low_price_recommendation(loc, tier, info.get("type") or "UNKNOWN", info.get("confirmations") or [],
                                    long_term_winner=not row.get("long_term_broken"))


def daily_holding_advice(act: str, row: Optional[Dict], market: str, report: Dict) -> tuple:
    """持股「今天」該做什麼（每天都有結論）。規則與回測一致：低檔加碼/集中度/不追大長紅每天檢查；動能買賣在月調窗口執行。"""
    window = report.get("strategy", {}).get("in_rebalance_window")
    nxt = report.get("strategy", {}).get("next_rebalance")
    sp = (row or {}).get("spike")
    if act in ("核心 ETF", "資料不足"):
        return "不用動", ""
    if act == "減碼":
        return "今天減碼", "集中度風險每天檢查，不必等月調。"
    if act == "低檔加碼":
        if sp:
            return "掛單等拉回", f"{sp['date']} 大漲 +{sp['gain_pct']}%，不追；掛 {sp.get('limit_price')}，{LIMIT_VALID_DAYS} 個交易日內有效。"
        return "今天可加碼", "在低檔區、部位不足：低檔訊號每天有效，可今天分批加碼。"
    if act == "加碼":
        if not window:
            return "月調日加碼", f"動能加碼排在月調日（{nxt}），月中不動：{FREQ_EVIDENCE}"
        if sp:
            return "等 3 天再加碼", f"{sp['date']} 大漲 +{sp['gain_pct']}%，等大長紅超過 {SPIKE_DAYS} 個交易日、仍在前段再加碼。"
        return "今天加碼", "月調窗口內、排名前段且部位不足。"
    if act == "賣出換股":
        if window:
            return "今天賣出換股", "月調窗口內：動能轉弱且長線破壞，賣出後換到低檔/動能名單。"
        return "月調日賣出", (f"轉弱：預計月調日（{nxt}）全部賣出換股（屆時仍跌出名單且長線破壞才賣）。"
                              f"不提前賣：{FREQ_EVIDENCE}")
    # 續抱
    if row:
        lep, close = row.get("low_entry_price"), row.get("close")
        if row.get("low_entry_watch") and lep and close and (close / lep - 1) * 100 <= NEAR_TRIGGER_PCT_HOLD:
            return "準備加碼", f"距低檔加碼價 {lep} 只差 {(close / lep - 1) * 100:.1f}%：跌到就是加碼點。"
        if sp:
            return "不用動", f"{sp['date']} 大漲 +{sp['gain_pct']}%：續抱，不追加。"
    return "不用動", "續抱，沒有新訊號。"


NEAR_TRIGGER_PCT_HOLD = 3.0
# 2026-10-09 換股頻率回測（低檔 70% + 動能 30%，2017 起，T-1 訊號 → T 開盤、成本 0.2%）：
# 月調 21 日 美股 CAGR 49.2% / 台股 47.1%；雙週 48.6% / 45.8%；每週 48.0% / 46.7%；月調＋每日出場 48.8% / 46.3%。
FREQ_EVIDENCE = "回測顯示動能換股改成每週或每日檢查，報酬反而較低（交易成本與來回洗），月調最佳。"


def evaluate_holdings(
    holdings: Iterable[Dict],
    report: Dict,
    fx_usd_twd: float = 32.0,
    extra_prices: Optional[Dict[str, pd.DataFrame]] = None,
    live: bool = False,
    cash_twd: float = 0.0,
) -> Dict:
    """holdings: [{symbol|ticker, cost, shares}]。回傳每檔動作與「新買進」清單、各市場目標金額。

    cash_twd：現金（台幣計），算集中度時分母用「持股＋現金」的總資產。
    live=True：現價/今日漲跌改用請求當下的即時報價（盤中看才是「今天」的漲跌）；動作判斷仍依每日收盤快照。

    動作：加碼（前 10 名且部位不足）、續抱、減碼（前 20 名但部位過重）、賣出換股（跌出前 20 名）、
         核心 ETF（不適用）、資料不足。
    """
    markets = report.get("markets", {})
    items = []
    for h in holdings:
        sym = to_yf(str(h.get("symbol") or h.get("ticker") or ""))
        if not sym:
            continue
        items.append({"symbol": sym, "cost": float(h.get("cost") or h.get("cost_basis") or 0),
                      "shares": float(h.get("shares") or 0), "market": market_of(sym)})

    # 價格與排名
    need_price = []
    for it in items:
        m = markets.get(it["market"], {})
        row = next((r for r in m.get("rows", []) if r["symbol"] == it["symbol"]), None)
        it["row"] = row
        if row is None:
            need_price.append(it["symbol"])
    prices = dict(extra_prices or {})
    missing = [s for s in need_price if s not in prices]
    if missing:
        prices.update(download_closes(missing, period="4y"))
        for s in [s for s in missing if s not in prices and s.endswith(".TW")]:
            alt = download_closes([s[:-3] + ".TWO"], period="4y")
            if alt:
                prices[s] = next(iter(alt.values()))
    quotes: Dict[str, Dict] = {}
    today_tpe = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    if live:
        try:
            from src.strategy.live_quote import live_quotes
            quotes = live_quotes(it["symbol"] for it in items)
        except Exception:
            quotes = {}
    for it in items:
        if it["row"] is None and it["symbol"] in prices:
            r = score_row(it["symbol"], prices[it["symbol"]])
            if r is None:
                f = prices[it["symbol"]]
                it["price"] = float(f["Close"].dropna().iloc[-1]) if len(f) else None
            else:
                rows = markets.get(it["market"], {}).get("rows", [])
                r["rank"] = _rank_for_score(r, rows)
                r["zone"] = zone_of(r["rank"], it["market"])
                r["outside_universe"] = True
                it["row"] = r
        if it.get("row"):
            it["price"] = it["row"]["close"]
        fx = 1.0 if it["market"] == "tw" else fx_usd_twd
        # 今日漲跌：有即時報價就用（現價 vs 昨收）；否則用快照最後一根收盤 vs 前一根
        lq = quotes.get(it["symbol"])
        if lq:
            it["price"], chg = lq["price"], lq["change_pct"]
            it["quote_as_of"], it["quote_source"] = lq.get("as_of"), lq.get("source")
            # 台股報價日期不是今天（台北）＝今天休市或還沒開盤：今日漲跌/損益記 0，不拿前一日的來算
            if it["market"] == "tw" and lq.get("as_of") and not lq["as_of"].startswith(today_tpe):
                chg, it["market_closed"] = 0.0, True
        else:
            chg = (it.get("row") or {}).get("day_change_pct")
            if chg is None and it["symbol"] in prices:
                c_ = prices[it["symbol"]]["Close"].dropna()
                chg = round((float(c_.iloc[-1]) / float(c_.iloc[-2]) - 1) * 100, 2) if len(c_) > 1 else None
        it["value_twd"] = (it.get("price") or 0) * it["shares"] * fx
        it["day_change_pct"] = chg
        it["day_pnl_twd"] = (it["value_twd"] - it["value_twd"] / (1 + chg / 100)) if chg is not None else 0.0

    total_twd = sum(it["value_twd"] for it in items) or 0.0
    conc_base = total_twd + max(0.0, float(cash_twd or 0))
    sleeve = {m: sum(it["value_twd"] for it in items if it["market"] == m and not is_etf(it["symbol"])) for m in ("us", "tw")}
    mshare = {m: markets.get(m, {}).get("momentum_share", ALLOCATION["momentum"]) for m in sleeve}
    target = {m: (sleeve[m] * mshare[m] / TOP_N[m] if sleeve[m] else 0.0) for m in sleeve}
    panic = panic_mode(report)
    low_target = {m: (sleeve[m] * ALLOCATION["lowentry"] / LOWENTRY_SLOTS if sleeve[m] else 0.0) for m in sleeve}

    out_rows = []
    for it in items:
        row, sym = it.get("row"), it["symbol"]
        price = it.get("price")
        pnl = ((price / it["cost"] - 1) * 100) if (price and it["cost"]) else None
        tgt = target[it["market"]]
        base = {
            "symbol": sym, "market": it["market"], "shares": it["shares"], "cost": it["cost"], "price": price,
            "pnl_pct": round(pnl, 1) if pnl is not None else None,
            "value_twd": round(it["value_twd"]), "weight_pct": round(it["value_twd"] / total_twd * 100, 1) if total_twd else 0,
            "rank": row.get("rank") if row else None, "score": row.get("score") if row else None,
            "ret_6m_pct": row.get("ret_6m_pct") if row else None,
            "target_twd": round(tgt) if tgt else None, "ignition": (row or {}).get("ignition"),
            "day_change_pct": it.get("day_change_pct"), "day_pnl_twd": round(it.get("day_pnl_twd") or 0),
            "quote_as_of": it.get("quote_as_of"), "quote_source": it.get("quote_source") or "daily_close",
            "market_closed": bool(it.get("market_closed")),
            "day_pnl_local": round((it.get("day_pnl_twd") or 0) / (1.0 if it["market"] == "tw" else fx_usd_twd), 2),
            "spike": (row or {}).get("spike"),
            "virattt": (row or {}).get("virattt"), "dd_52w_pct": (row or {}).get("dd_52w_pct"),
        }
        if is_etf(sym):
            act, why = "核心 ETF", "ETF 屬核心部位，不套用個股動能輪動；長期持有即可。"
            base["rank"] = base["score"] = None
        elif row is None:
            act, why = "資料不足", "上市未滿半年或抓不到報價，暫不評分。"
        elif conc_base and it["value_twd"] / conc_base * 100 > CONCENTRATION_PCT:
            act = "減碼"
            trim_twd = it["value_twd"] - conc_base * CONCENTRATION_PCT / 100
            px_twd = (price or 0) * (1.0 if it["market"] == "tw" else fx_usd_twd)
            trim_sh = min(it["shares"], math.ceil(trim_twd / px_twd)) if px_twd else None
            base["trim_twd"], base["trim_shares"] = round(trim_twd), trim_sh
            keep = it["shares"] - (trim_sh or 0)
            lots = f"（約 {trim_sh / 1000:.1f} 張）" if it["market"] == "tw" and trim_sh and trim_sh >= 1000 else ""
            why = (f"單檔佔總資產 {it['value_twd'] / conc_base:.0%}，超過 {CONCENTRATION_PCT:.0f}% 集中度上限："
                   f"賣 {trim_sh:,.0f} 股{lots}、約 NT${trim_twd:,.0f}（持股的 {trim_sh / it['shares']:.0%}），"
                   f"留 {keep:,.0f} 股、降到約 {CONCENTRATION_PCT:.0f}%——不是全賣；賣出資金轉到低檔布局/動能名單。")
        elif (row.get("low_entry") and it["value_twd"] < low_target[it["market"]] * UNDERWEIGHT_RATIO and not row.get("outside_universe")
              and (low_view(row, report) or {}).get("recommendation") not in ("BUY", "BUY_STAGED")):
            lv = low_view(row, report) or {}
            act = "續抱"
            why = (f"價格跌深（距高點 {_pct(row.get('dd_52w_pct'), False)}），但{lv.get('label', '')}：{lv.get('why', '')}"
                   "不加碼。")
        elif row.get("low_entry") and it["value_twd"] < low_target[it["market"]] * UNDERWEIGHT_RATIO and not row.get("outside_universe"):
            lt = low_target[it["market"]]
            act = "低檔加碼"
            why = (f"{(low_view(row, report) or {}).get('label', '')}｜長線贏家（3 年 {_pct(row.get('ret_3y_pct'))}）已自 52 週高點回落 {_pct(abs(row['dd_52w_pct']) if row.get('dd_52w_pct') is not None else None, False)}，進入低檔布局區；"
                   f"部位僅目標的 {it['value_twd'] / lt:.0%}，可分批加碼到約 NT${lt:,.0f}，持有 {LOWENTRY_HOLD_MONTHS} 個月。")
        elif row["rank"] <= TOP_N[it["market"]] and tgt and it["value_twd"] < tgt * UNDERWEIGHT_RATIO and not row.get("outside_universe"):
            act = "加碼"
            why = f"動能排名第 {row['rank']} 名（前 {TOP_N[it['market']]} 名買進區），部位僅目標的 {it['value_twd'] / tgt:.0%}，加碼至約 NT${tgt:,.0f}。"
        elif row["rank"] > KEEP_N[it["market"]] and row.get("long_term_broken") and panic:
            act = "續抱"
            why = (f"動能排名第 {row['rank']} 名、長線趨勢也破壞，原本該換股；但市場處於極度恐懼"
                   f"（恐懼貪婪 < {PANIC_NO_SELL_FG:.0f}）——回測顯示恐慌時不賣、等情緒回穩再換，報酬較高、回撤較小。")
        elif row["rank"] > KEEP_N[it["market"]] and row.get("long_term_broken"):
            act = "賣出換股"
            why = (f"動能排名第 {row['rank']} 名（已跌出前 {KEEP_N[it['market']]} 名），且 {LOWENTRY_LT_YEARS} 年報酬 "
                   f"{_pct(row.get('ret_3y_pct'))}（長線趨勢已破壞）——兩條策略都不支持續抱，建議月初換到低檔布局或動能名單。")
        else:
            act = "續抱"
            if row.get("low_entry"):
                lv = low_view(row, report) or {}
                if lv.get("recommendation") in ("BUY", "BUY_STAGED"):
                    why = (f"{lv.get('label')}（距 52 週高點 {_pct(row.get('dd_52w_pct'), False)}、3 年 {_pct(row.get('ret_3y_pct'))}），"
                           f"部位已足，續抱 {LOWENTRY_HOLD_MONTHS} 個月。")
                else:
                    why = (f"價格跌深（距 52 週高點 {_pct(row.get('dd_52w_pct'), False)}），但{lv.get('label') or '不屬於可買的低檔'}："
                           f"{lv.get('why') or ''}續抱、不加碼。")
            elif row["rank"] <= KEEP_N[it["market"]]:
                why = f"動能排名第 {row['rank']} 名（前 {KEEP_N[it['market']]} 名內），續抱。"
            else:
                why = (f"動能排名第 {row['rank']} 名偏弱，但 {LOWENTRY_LT_YEARS} 年長線趨勢仍向上（{_pct(row.get('ret_3y_pct'))}）、"
                       f"距高點 {_pct(row.get('dd_52w_pct'), False)}——長線續抱、不加碼；"
                       + (f"跌到 {row.get('low_entry_price')}（回落 30%）才是低檔加碼點。" if row.get("low_entry_price") else ""))
        if row and row.get("outside_universe") and act not in ("核心 ETF", "資料不足"):
            why += " 註：此股不在 AI 科技股池，排名為換算參考；策略不會新買或加碼它。"
        if act in ("低檔加碼", "加碼"):
            why += no_chase_note(row)
        if base["ignition"] and act in ("續抱", "加碼"):
            why += f" 🔥 近 {base['ignition']['ignition_days_ago']} 日爆量長紅點火。"
        today, today_why = daily_holding_advice(act, row, it["market"], report)
        if act == "減碼" and base.get("trim_shares"):
            today = f"今天減碼 {base['trim_shares']:,.0f} 股"
            today_why = f"只賣 {base['trim_shares'] / it['shares']:.0%}（約 NT${base['trim_twd']:,.0f}），降到總資產 {CONCENTRATION_PCT:.0f}%；集中度每天檢查。"
        out_rows.append({**base, "action": act, "reason": why, "today": today, "today_reason": today_why})

    held = {it["symbol"] for it in items}
    new_buys = {}
    for m in ("us", "tw"):
        rows = markets.get(m, {}).get("rows", [])
        new_buys[m] = [
            {**{k: r.get(k) for k in ("symbol", "name", "rank", "score", "close", "ret_6m_pct", "ignition", "spike")},
             "target_twd": round(target[m]) if target[m] else None}
            for r in rows[:TOP_N[m]] if r["symbol"] not in held
        ]
    low_buys = {}
    for m in ("us", "tw"):
        mk = markets.get(m, {})
        by = {r["symbol"]: r for r in mk.get("rows", [])}
        low_buys[m] = [
            {**{k: by[s_].get(k) for k in ("symbol", "name", "rank", "close", "dd_52w_pct", "ret_3y_pct", "high_52w", "spike",
                                           "bounce_20d_pct", "quality_tier")},
             "low_label": (low_view(by[s_], report) or {}).get("label"),
             "target_twd": round(low_target[m] * (0.5 if (low_view(by[s_], report) or {}).get("recommendation") == "BUY_STAGED" else 1))
             if low_target[m] else None}
            for s_ in mk.get("low_entry", []) if s_ in by and s_ not in held
            and (low_view(by[s_], report) or {}).get("recommendation") in ("BUY", "BUY_STAGED")
        ][:LOWENTRY_SLOTS]

    order = {"賣出換股": 0, "減碼": 1, "低檔加碼": 2, "加碼": 3, "續抱": 4, "核心 ETF": 5, "資料不足": 6}
    out_rows.sort(key=lambda r: (order.get(r["action"], 9), -(r["value_twd"] or 0)))
    return {
        "generated_at": report.get("generated_at"),
        "next_rebalance": report.get("strategy", {}).get("next_rebalance"),
        "in_rebalance_window": report.get("strategy", {}).get("in_rebalance_window"),
        "fx_usd_twd": round(fx_usd_twd, 3),
        "live_quotes": bool(quotes),
        "quote_as_of": {m: max((it.get("quote_as_of") for it in items if it["market"] == m and it.get("quote_as_of")), default=None)
                        for m in ("us", "tw")},
        "market_closed": {m: any(it.get("market_closed") for it in items if it["market"] == m) for m in ("us", "tw")},
        "total_twd": round(total_twd),
        "day_pnl_twd": round(sum(it.get("day_pnl_twd") or 0 for it in items)),
        "day_change_pct": round(sum(it.get("day_pnl_twd") or 0 for it in items) / (total_twd - sum(it.get("day_pnl_twd") or 0 for it in items)) * 100, 2)
                          if total_twd else None,
        "day_pnl_by_market_twd": {m: round(sum(it.get("day_pnl_twd") or 0 for it in items if it["market"] == m)) for m in ("us", "tw")},
        "sleeve_twd": {m: round(v) for m, v in sleeve.items()},
        "target_per_name_twd": {m: round(v) for m, v in target.items()},
        "low_entry_target_twd": {m: round(v) for m, v in low_target.items()},
        "allocation": ALLOCATION,
        "momentum_share": mshare,
        "panic_no_sell": panic,
        "holdings": out_rows,
        "new_buys": new_buys,
        "low_entry_buys": low_buys,
    }


def usd_twd() -> float:
    try:
        import yfinance as yf
        px = yf.Ticker("TWD=X").history(period="5d")["Close"].dropna()
        return float(px.iloc[-1]) if len(px) else 32.0
    except Exception:
        return 32.0


# ---------------------------------------------------------------------------
# 個股查詢（網站「個股分析」）
# ---------------------------------------------------------------------------
def lookup_symbols(tickers: Iterable[str], report: Dict) -> List[Dict]:
    """查任意代號在策略中的排名與結論（不在股票池者以分數換算等效排名），附近一年收盤供畫圖。"""
    syms = [to_yf(t) for t in tickers if str(t).strip()]
    prices = download_closes(syms, period="4y")
    for s in [s for s in syms if s not in prices and s.endswith(".TW")]:
        alt = download_closes([s[:-3] + ".TWO"], period="4y")
        if alt:
            prices[s] = next(iter(alt.values()))
    out = []
    for sym in syms:
        m = market_of(sym)
        mk = report.get("markets", {}).get(m, {})
        rows = mk.get("rows", [])
        f = prices.get(sym)
        row = next((dict(r) for r in rows if r["symbol"] == sym), None)
        if row is None and f is not None:
            row = score_row(sym, f)
            if row:
                row["rank"] = _rank_for_score(row, rows)
                row["zone"] = zone_of(row["rank"], m)
                row["outside_universe"] = True
        keep = KEEP_N[m]
        if is_etf(sym):
            verdict, detail = "核心 ETF", "ETF 屬核心部位，不套用個股動能輪動；適合長期持有。"
        elif row is None:
            verdict, detail = "資料不足", "抓不到報價或上市未滿半年，暫不評分。"
        elif row.get("low_entry") and not row.get("outside_universe"):
            lv = low_view(row, report, mk.get("universe_size")) or {}
            verdict = lv.get("label") or "低檔觀察（不直接買）"
            detail = (f"價格位置：自 52 週高點 {row.get('high_52w')} 回落 {_pct(row.get('dd_52w_pct'), False)}（DEEPLY_DISCOUNTED）；"
                      f"排名第 {row['rank']}/{mk.get('universe_size')}（Tier {lv.get('tier')}）、回撤分類 {lv.get('drawdown_type')}。"
                      + (lv.get("why") or "")
                      + (f"持有 {LOWENTRY_HOLD_MONTHS} 個月。" if lv.get("recommendation") in ("BUY", "BUY_STAGED") else ""))
        elif row["zone"] == "buy" and row.get("outside_universe"):
            verdict, detail = "持有續抱，不新買", f"不在 AI 科技股池；換算排名第 {row['rank']} 名，動能強但策略只買池內標的。已持有可續抱。"
        elif row["zone"] == "buy":
            verdict, detail = "可買進", f"動能排名第 {row['rank']} 名，在前 {TOP_N[m]} 名買進區；新資金可等權買進，每月初再檢查。"
        elif row["zone"] == "hold":
            verdict, detail = "持有續抱，不新買", f"排名第 {row['rank']} 名，在續抱區（{TOP_N[m] + 1}~{keep} 名）：已持有可續抱，沒持有則等它進前 {TOP_N[m]} 名。"
        else:
            if row.get("long_term_broken"):
                verdict, detail = "不買／月初換股", f"動能排名第 {row['rank']} 名且 3 年長線趨勢已破壞；不建議買進。"
            else:
                lep = row.get("low_entry_price")
                verdict, detail = "等低檔", (f"動能排名第 {row['rank']} 名偏弱、尚未跌到低檔區（距高點 {_pct(row.get('dd_52w_pct'), False)}）；"
                                            + (f"跌到 {lep} 以下（回落 30%）才進入低檔布局區。已持有者長線續抱。" if lep else "觀望。"))
        if verdict in ("可買進",) or verdict.startswith(("低檔布局可買", "低檔分批")):
            detail += no_chase_note(row)
        closes = []
        if f is not None:
            c = f["Close"].dropna()
            closes = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 2)} for d, v in c.items()]
        out.append({
            "symbol": sym, "market": m, "universe_size": mk.get("universe_size"), "keep_n": keep, "top_n": TOP_N[m],
            "row": row, "verdict": verdict, "detail": detail, "closes": closes,
        })
    return out
