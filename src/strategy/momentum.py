"""單一策略：6 個月風險調整動能（Sharpe 動能）排名輪動。2026-10-06 全面改版。

為什麼換掉舊框架：舊的「綜合分數」是一堆手調規則（RSI 加減分、回檔加分、大師 agent 投票…），
自己的檢討紀錄就寫著與未來報酬的相關性近乎 0。這次改成先做因子研究、只留回測驗證過的訊號：

  因子研究（S&P 500 + 台股權值/AI 共 570 檔、2017~2026、分前後兩段樣本外驗證）：
  - 台股：6 個月風險調整動能 rank IC 0.07（20 日）/ 0.115（60 日），前後兩段都穩定，是所有因子中最強。
  - 美股：多數價格因子很弱，12 個月動能前後兩段甚至正負翻轉；Sharpe 動能是最「穩定」的一個。

  投組回測（月調、每次換股成本 0.2%、持有前 10 名）：
  - 台股 跌出前 30 名才賣：年化 46.1%、最大回撤 -31%、Sharpe 1.65；0050 同期 24.7% / -34% / 1.20
  - 美股 跌出前 100 名才賣：年化 33.1%、最大回撤 -34%、Sharpe 1.19；VOO 同期 15.3% / -34% / 0.86
    （更嚴的「跌出前 20 名就賣」年化 36.0%，但換手多一倍；預設採較寬緩衝，少賣少錯殺）
  兩段樣本（2017-21、2022-26）都勝過大盤。注意：股票池用「現在的」成分股，有倖存者偏差，
  真實超額會比回測小。

規則（網站、每日報告、自動提醒、回測全部共用這一份）：
  - 分數 = 近 126 個交易日日報酬的平均 / 標準差（年化顯示）。
  - 每市場持有排名前 TOP_N 檔、等權；每月第一個交易日檢查。
  - 持股跌出前 KEEP_N[市場] 名才賣出換股（美股 100、台股 30 的緩衝帶）。
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
TOP_N = 10
KEEP_N = {"us": 100, "tw": 30}
UNDERWEIGHT_RATIO = 0.7   # 部位 < 目標 × 0.7 → 加碼
CONCENTRATION_PCT = 20.0  # 單檔 > 總資產 20% → 減碼（集中度風險）
TPE = timezone(timedelta(hours=8))

US_ETFS = {
    "VOO", "SPY", "IVV", "VTI", "QQQ", "QQQM", "DIA", "IWM", "VT", "VXUS", "SCHD", "VIG", "VUG",
    "SOXX", "SMH", "XLK", "TLT", "IEF", "SHY", "BND", "AGG", "GLD", "IAU", "SGOV", "BIL", "TQQQ", "SOXL",
}
# 美股：S&P 500 以外但屬 AI 主線的個股，一併排名。
US_EXTRAS = [
    "TSM", "ASML", "ARM", "CEG", "VST", "TLN", "NBIS", "CRWV", "SNDK", "ALAB", "CRDO", "LITE", "BE",
    "OKLO", "APP", "HOOD", "COIN", "PLTR", "SMCI", "COHR", "MRVL", "VRT", "ANET", "DELL", "NRG", "GEV",
]
TW_EXTRAS = ["2379", "3443", "6669", "2376", "2377", "3037", "3008", "2327", "2345", "3017", "3661"]


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


def sp500_symbols() -> List[str]:
    try:
        import requests
        csv = requests.get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            timeout=30,
        ).text
        syms = [str(s).replace(".", "-") for s in pd.read_csv(io.StringIO(csv))["Symbol"]]
        if len(syms) > 400:
            return syms
    except Exception as e:
        print(f"[strategy] S&P 500 清單下載失敗，改用備援：{e}")
    from src.sp500_daily import fetch_sp500_constituents
    return [c.yf_symbol for c in fetch_sp500_constituents()]


def universe(market: str) -> List[str]:
    if market == "us":
        syms = sp500_symbols() + US_EXTRAS
    else:
        from src.pipeline.daily_report import TW_UNIVERSE
        syms = [to_yf(t) for t in list(TW_UNIVERSE) + TW_EXTRAS]
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
    return "buy" if rank <= TOP_N else ("hold" if rank <= KEEP_N[market] else "out")


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
        "threshold_top": rows[TOP_N - 1]["score"] if len(rows) >= TOP_N else None,
        "keep_n": KEEP_N[market],
        "threshold_keep": rows[KEEP_N[market] - 1]["score"] if len(rows) >= KEEP_N[market] else None,
        "rows": rows,
    }


def _names(market: str) -> Dict[str, str]:
    if market != "tw":
        return {}
    try:
        from src.pipeline.daily_report import TW_NAMES
        return dict(TW_NAMES)
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
            "rule": f"每市場持有排名前 {TOP_N} 名（等權），每月初檢查；跌出前 {KEEP_N['us']} 名（美股）/ {KEEP_N['tw']} 名（台股）才賣出換股。",
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
    target = {m: (sleeve[m] / TOP_N if sleeve[m] else 0.0) for m in sleeve}

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
                   f"已跌出前 {KEEP_N[it['market']]} 名；依策略於月初賣出，資金換到前 {TOP_N} 名的新買進標的。")
        elif total_twd and it["value_twd"] / total_twd * 100 > CONCENTRATION_PCT:
            act = "減碼"
            why = (f"排名第 {row['rank']} 名仍在名單內，但單檔佔總資產 {it['value_twd'] / total_twd:.0%}，"
                   f"超過 {CONCENTRATION_PCT:.0f}% 集中度上限；減碼到 {CONCENTRATION_PCT:.0f}% 以下，資金分散到新買進標的。")
        elif row["rank"] <= TOP_N and tgt and it["value_twd"] < tgt * UNDERWEIGHT_RATIO:
            act = "加碼"
            why = f"排名第 {row['rank']} 名（前 {TOP_N} 名買進區），部位僅目標的 {it['value_twd'] / tgt:.0%}，加碼至每檔約 NT${tgt:,.0f}。"
        else:
            act = "續抱"
            why = (f"排名第 {row['rank']} 名，" + ("在買進區、部位已達目標。" if row["rank"] <= TOP_N
                                                 else f"在續抱區（{TOP_N + 1}~{KEEP_N[it['market']]} 名），不加碼也不賣。"))
            if tgt and it["value_twd"] > tgt * 1.6:
                why += f" 部位已是每檔目標的 {it['value_twd'] / tgt:.1f} 倍，新資金優先買其他名單股。"
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
            for r in rows[:TOP_N] if r["symbol"] not in held
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
