"""SEC EDGAR XBRL：point-in-time 基本面（以「申報日 filed」為準，不是期末日）。

用途：重建「當時市場已知」的 TTM 營收、TTM 淨利、流通股數 → 歷史 P/S、P/E，供估值回測。
規範：SEC 要求自動程式在 User-Agent 附聯絡資訊 → 由環境變數 SEC_USER_AGENT 提供（不寫進 repo）。
速率：SEC 上限每秒 10 次，這裡每次請求間隔 ≥ 0.15 秒；結果快取在 .cache/sec/（已 gitignore）。
限制：只涵蓋向 SEC 申報 10-K/10-Q 的美國公司；外國發行人（20-F，如 TSM/ASML/ARM）與台股不支援。
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".cache", "sec")
REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"]
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
_last_call = [0.0]


def _ua() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError("未設定 SEC_USER_AGENT（需含聯絡 email，例：'ai-hedge-fund-tw research you@example.com'）")
    return ua


def _get(url: str) -> dict:
    wait = 0.15 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"}, timeout=60)
    _last_call[0] = time.time()
    r.raise_for_status()
    return r.json()


def _cached(name: str, url: str, max_age_days: int) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_days * 86400:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    data = _get(url)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def cik_map() -> Dict[str, int]:
    data = _cached("company_tickers.json", "https://www.sec.gov/files/company_tickers.json", 30)
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


def company_facts(ticker: str) -> Optional[dict]:
    cik = cik_map().get(ticker.upper())
    if cik is None:
        return None
    return _cached(f"facts_{cik}.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", 7)


def _facts(cf: dict, taxonomy: str, tags: List[str], unit: str) -> pd.DataFrame:
    """合併多個同義 tag（依優先序；同一期間保留優先 tag）。欄位：start,end,val,filed,form。"""
    frames = []
    for pri, t in enumerate(tags):
        items = cf.get("facts", {}).get(taxonomy, {}).get(t, {}).get("units", {}).get(unit, [])
        for it in items:
            if it.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue
            frames.append({"start": it.get("start"), "end": it.get("end"), "val": it.get("val"),
                           "filed": it.get("filed"), "form": it.get("form"), "pri": pri})
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form", "pri"])
    df = pd.DataFrame(frames)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["end", "filed", "val"])
    return df.sort_values(["pri", "filed"])


def ttm_series(flow: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """每個日期 d：只用 filed ≤ d 的資料，TTM = 最近年報 + 之後的年初至今 − 去年同期年初至今。"""
    flow = flow.dropna(subset=["start"]).copy()
    if flow.empty:
        return pd.Series(index=dates, dtype=float)
    flow["days"] = (flow["end"] - flow["start"]).dt.days
    out, cache = {}, {}
    filings = sorted(flow["filed"].unique())
    for f in filings:
        known = flow[flow["filed"] <= f]
        ann = known[(known["days"] >= 350) & (known["days"] <= 380)]
        if ann.empty:
            continue
        a = ann.sort_values(["end", "pri"], ascending=[False, True]).iloc[0]
        ttm = float(a["val"])
        ytd = known[(known["start"] - a["end"]).dt.days.between(0, 7) & (known["days"] < 350)]
        if not ytd.empty:
            y = ytd.sort_values(["end", "pri"], ascending=[False, True]).iloc[0]
            prior = known[((known["end"] - (y["end"] - pd.DateOffset(years=1))).dt.days.abs() <= 10)
                          & ((known["days"] - y["days"]).abs() <= 10)]
            if not prior.empty:
                p = prior.sort_values("pri").iloc[0]
                ttm = ttm + float(y["val"]) - float(p["val"])
        cache[pd.Timestamp(f)] = ttm
    s = pd.Series(cache).sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def shares_series(cf: dict, dates: pd.DatetimeIndex, splits: Optional[pd.Series] = None) -> pd.Series:
    """申報時的流通股數換算成「今日分割後」單位（與 yfinance 還原分割後的股價一致）。"""
    items = cf.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    rows = [(pd.Timestamp(i["filed"]), float(i["val"])) for i in items if i.get("filed") and i.get("val")]
    if not rows:
        return pd.Series(index=dates, dtype=float)
    s = pd.DataFrame(rows, columns=["filed", "val"]).groupby("filed")["val"].sum().sort_index()
    if splits is not None and len(splits):
        sp = pd.Series(splits.values, index=pd.to_datetime(splits.index).tz_localize(None)
                       if getattr(splits.index, "tz", None) else pd.to_datetime(splits.index))
        s = pd.Series([v * float(sp[sp.index > f].prod()) for f, v in s.items()], index=s.index)
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def pit_valuation(ticker: str, close: pd.Series, splits: Optional[pd.Series] = None) -> Optional[pd.DataFrame]:
    """回傳每日 point-in-time 的 market_cap、ttm_revenue、ttm_net_income、ps、pe。
    close 需為「分割還原後」價格（yfinance 預設）；splits 為 yfinance Ticker.splits（未提供時自動抓）。"""
    if splits is None:
        try:
            import yfinance as yf
            splits = yf.Ticker(ticker).splits
        except Exception:
            splits = None
    cf = company_facts(ticker)
    if not cf:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None) if getattr(close.index, "tz", None) else pd.DatetimeIndex(close.index)
    close = pd.Series(close.values, index=idx)
    rev = ttm_series(_facts(cf, "us-gaap", REVENUE_TAGS, "USD"), idx)
    ni = ttm_series(_facts(cf, "us-gaap", NET_INCOME_TAGS, "USD"), idx)
    sh = shares_series(cf, idx, splits)
    mcap = close * sh
    df = pd.DataFrame({"market_cap": mcap, "ttm_revenue": rev, "ttm_net_income": ni})
    df["ps"] = df["market_cap"] / df["ttm_revenue"].where(df["ttm_revenue"] > 0)
    df["pe"] = df["market_cap"] / df["ttm_net_income"].where(df["ttm_net_income"] > 0)
    return df
