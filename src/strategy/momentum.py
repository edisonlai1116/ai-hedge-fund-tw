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
CONCENTRATION_PCT = 20.0  # 單檔 > 總資產 20% → 減碼（集中度風險）
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
    ign = _ignition(f)
    if ign:
        row["ignition"] = {k: ign.get(k) for k in ("ignition_days_ago", "ignition_gain_pct", "ignition_volume_ratio", "ignition_low")}
    return row


def zone_of(rank: int, market: str) -> str:
    return "buy" if rank <= TOP_N[market] else ("hold" if rank <= KEEP_N[market] else "out")


def rank_market(market: str, price_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
    syms = universe(market)
    price_map = price_map if price_map is not None else download_closes(syms)
    rows = [r for s in syms if s in price_map for r in [score_row(s, price_map[s])] if r]
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["zone"] = zone_of(i, market)
    names = _names(market)
    for r in rows:
        r["name"] = names.get(r["symbol"].split(".")[0], "")
    return {
        "universe_size": len(rows),
        "top_n": TOP_N[market],
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


def build_strategy_report() -> Dict:
    now = datetime.now(TPE)
    markets = {}
    for m in ("us", "tw"):
        try:
            markets[m] = rank_market(m)
        except Exception as e:
            print(f"[strategy] {m} 排名失敗：{e}")
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "strategy": {
            "name": "Sharpe 動能輪動",
            "lookback_days": LOOKBACK, "top_n": TOP_N, "keep_n": KEEP_N,
            "rule": (f"AI 科技股池依動能排名：美股持有前 {TOP_N['us']} 名、跌出前 {KEEP_N['us']} 名才賣；"
                     f"台股持有前 {TOP_N['tw']} 名、跌出前 {KEEP_N['tw']} 名才賣。等權、每月初檢查。"),
            "next_rebalance": next_rebalance(now.date()),
            "in_rebalance_window": is_rebalance_window(now.date()),
        },
        "markets": markets,
        "disclaimer": "規則化訊號，回測有倖存者偏差，非投資建議。",
    }


# ---------------------------------------------------------------------------
# 持股評估（網站「我的持股」與自動提醒共用）
# ---------------------------------------------------------------------------
def _rank_for_score(score: float, rows: List[Dict]) -> int:
    """非股票池個股：以分數插入股票池排名，得到等效名次。"""
    return 1 + sum(1 for r in rows if r["score"] > score)


def evaluate_holdings(
    holdings: Iterable[Dict],
    report: Dict,
    fx_usd_twd: float = 32.0,
    extra_prices: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """holdings: [{symbol|ticker, cost, shares}]。回傳每檔動作與「新買進」清單、各市場目標金額。

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
        prices.update(download_closes(missing))
        for s in [s for s in missing if s not in prices and s.endswith(".TW")]:
            alt = download_closes([s[:-3] + ".TWO"])
            if alt:
                prices[s] = next(iter(alt.values()))
    for it in items:
        if it["row"] is None and it["symbol"] in prices:
            r = score_row(it["symbol"], prices[it["symbol"]])
            if r is None:
                f = prices[it["symbol"]]
                it["price"] = float(f["Close"].dropna().iloc[-1]) if len(f) else None
            else:
                rows = markets.get(it["market"], {}).get("rows", [])
                r["rank"] = _rank_for_score(r["score"], rows)
                r["zone"] = zone_of(r["rank"], it["market"])
                r["outside_universe"] = True
                it["row"] = r
        if it.get("row"):
            it["price"] = it["row"]["close"]
        fx = 1.0 if it["market"] == "tw" else fx_usd_twd
        it["value_twd"] = (it.get("price") or 0) * it["shares"] * fx

    total_twd = sum(it["value_twd"] for it in items) or 0.0
    sleeve = {m: sum(it["value_twd"] for it in items if it["market"] == m and not is_etf(it["symbol"])) for m in ("us", "tw")}
    target = {m: (sleeve[m] / TOP_N[m] if sleeve[m] else 0.0) for m in sleeve}

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
        }
        if is_etf(sym):
            act, why = "核心 ETF", "ETF 屬核心部位，不套用個股動能輪動；長期持有即可。"
            base["rank"] = base["score"] = None
        elif row is None:
            act, why = "資料不足", "上市未滿半年或抓不到報價，暫不評分。"
        elif row["rank"] > KEEP_N[it["market"]]:
            act = "賣出換股"
            why = (f"動能排名第 {row['rank']} 名（{it['market'].upper()} 共 {markets.get(it['market'], {}).get('universe_size', '?')} 檔），"
                   f"已跌出前 {KEEP_N[it['market']]} 名；依策略於月初賣出，資金換到前 {TOP_N[it['market']]} 名的新買進標的。")
        elif total_twd and it["value_twd"] / total_twd * 100 > CONCENTRATION_PCT:
            act = "減碼"
            why = (f"排名第 {row['rank']} 名仍在名單內，但單檔佔總資產 {it['value_twd'] / total_twd:.0%}，"
                   f"超過 {CONCENTRATION_PCT:.0f}% 集中度上限；減碼到 {CONCENTRATION_PCT:.0f}% 以下，資金分散到新買進標的。")
        elif row["rank"] <= TOP_N[it["market"]] and tgt and it["value_twd"] < tgt * UNDERWEIGHT_RATIO and not row.get("outside_universe"):
            act = "加碼"
            why = f"排名第 {row['rank']} 名（前 {TOP_N[it['market']]} 名買進區），部位僅目標的 {it['value_twd'] / tgt:.0%}，加碼至每檔約 NT${tgt:,.0f}。"
        else:
            act = "續抱"
            why = (f"排名第 {row['rank']} 名，" + ("在買進區、部位已達目標。" if row["rank"] <= TOP_N[it["market"]]
                                                 else f"在續抱區（{TOP_N[it['market']] + 1}~{KEEP_N[it['market']]} 名），不加碼也不賣。"))
            if tgt and it["value_twd"] > tgt * 1.6:
                why += f" 部位已是每檔目標的 {it['value_twd'] / tgt:.1f} 倍，新資金優先買其他名單股。"
        if row and row.get("outside_universe") and act not in ("核心 ETF", "資料不足"):
            why += " 註：此股不在 AI 科技股池，排名為換算參考；策略不會新買或加碼它。"
        if base["ignition"] and act in ("續抱", "加碼"):
            why += f" 🔥 近 {base['ignition']['ignition_days_ago']} 日爆量長紅點火。"
        out_rows.append({**base, "action": act, "reason": why})

    held = {it["symbol"] for it in items}
    new_buys = {}
    for m in ("us", "tw"):
        rows = markets.get(m, {}).get("rows", [])
        new_buys[m] = [
            {**{k: r.get(k) for k in ("symbol", "name", "rank", "score", "close", "ret_6m_pct", "ignition")},
             "target_twd": round(target[m]) if target[m] else None}
            for r in rows[:TOP_N[m]] if r["symbol"] not in held
        ]

    order = {"賣出換股": 0, "減碼": 1, "加碼": 2, "續抱": 3, "核心 ETF": 4, "資料不足": 5}
    out_rows.sort(key=lambda r: (order.get(r["action"], 9), -(r["value_twd"] or 0)))
    return {
        "generated_at": report.get("generated_at"),
        "next_rebalance": report.get("strategy", {}).get("next_rebalance"),
        "in_rebalance_window": report.get("strategy", {}).get("in_rebalance_window"),
        "fx_usd_twd": round(fx_usd_twd, 3),
        "total_twd": round(total_twd),
        "sleeve_twd": {m: round(v) for m, v in sleeve.items()},
        "target_per_name_twd": {m: round(v) for m, v in target.items()},
        "holdings": out_rows,
        "new_buys": new_buys,
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
    prices = download_closes(syms, period="1y")
    for s in [s for s in syms if s not in prices and s.endswith(".TW")]:
        alt = download_closes([s[:-3] + ".TWO"], period="1y")
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
                row["rank"] = _rank_for_score(row["score"], rows)
                row["zone"] = zone_of(row["rank"], m)
                row["outside_universe"] = True
        keep = KEEP_N[m]
        if is_etf(sym):
            verdict, detail = "核心 ETF", "ETF 屬核心部位，不套用個股動能輪動；適合長期持有。"
        elif row is None:
            verdict, detail = "資料不足", "抓不到報價或上市未滿半年，暫不評分。"
        elif row["zone"] == "buy" and row.get("outside_universe"):
            verdict, detail = "持有續抱，不新買", f"不在 AI 科技股池；換算排名第 {row['rank']} 名，動能強但策略只買池內標的。已持有可續抱。"
        elif row["zone"] == "buy":
            verdict, detail = "可買進", f"動能排名第 {row['rank']} 名，在前 {TOP_N[m]} 名買進區；新資金可等權買進，每月初再檢查。"
        elif row["zone"] == "hold":
            verdict, detail = "持有續抱，不新買", f"排名第 {row['rank']} 名，在續抱區（{TOP_N[m] + 1}~{keep} 名）：已持有可續抱，沒持有則等它進前 {TOP_N[m]} 名。"
        else:
            verdict, detail = "不買／月初換股", f"排名第 {row['rank']} 名，已在前 {keep} 名之外；不建議買進，已持有者依策略於月初賣出換股。"
        closes = []
        if f is not None:
            c = f["Close"].dropna()
            closes = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 2)} for d, v in c.items()]
        out.append({
            "symbol": sym, "market": m, "universe_size": mk.get("universe_size"), "keep_n": keep, "top_n": TOP_N[m],
            "row": row, "verdict": verdict, "detail": detail, "closes": closes,
        })
    return out
