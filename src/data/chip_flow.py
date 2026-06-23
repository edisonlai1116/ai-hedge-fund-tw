"""籌碼面 / 法人動向因子（免 API Key）。

台股（上市）：
  - TWSE 開放資料 T86（三大法人買賣超），一次抓全市場、快取當日，50 檔台股共用 1 request。
  - 以欄位名稱比對（不寫死 index）：TWSE 偶爾改欄序不影響解析。

美股：
  - 無公開每日法人流 → 用 Chaikin Money Flow (CMF-20) 代理資金方向，零額外抓取。

任何失敗安全降級回 None（呼叫端以中性 50 代入）。純函式可離線測試。
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

# --- 當日快取（記憶體，不寫磁碟，process 重啟即清） ------------------
_TW_CACHE: Dict[str, dict] = {}   # key = "YYYYMMDD", value = {stock_code: {...}}


# ============================================================
# 純函式（可離線單元測試）
# ============================================================

def score_from_net_ratio(net_ratio_pct: float) -> int:
    """淨買超佔量比（外資+投信合計）映射到 0-100 分。
    +10% → 100, 0% → 50, -10% → 0（線性夾擠）。
    """
    return int(max(0, min(100, 50 + net_ratio_pct * 5.0)))


def score_from_streak(streak: int) -> int:
    """連買/連賣天數映射到 0-100。
    連買 +5 → 90, 0 → 50, 連賣 -5 → 10（線性夾擠）。
    """
    return int(max(0, min(100, 50 + streak * 8.0)))


def blend_tw_score(net_ratio_pct: float, streak: int) -> int:
    """合併淨買超佔量比與連買/賣天數（各 50%）。"""
    s1 = score_from_net_ratio(net_ratio_pct)
    s2 = score_from_streak(streak)
    return int(round((s1 + s2) / 2))


def chip_label(score: int) -> str:
    if score >= 65:
        return "法人偏多"
    if score < 45:
        return "法人偏空"
    return "法人中性"


def compute_cmf(df, period: int = 20) -> float:
    """Chaikin Money Flow (CMF)，回傳最新值 -1~+1。df 需有 Close/High/Low/Volume。
    資料不足或計算失敗回 0.0（中性）。
    """
    if not _PANDAS_OK:
        return 0.0
    try:
        close = df["Close"].astype(float)
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)
        vol   = df["Volume"].astype(float)
        denom = high - low
        mfm = ((close - low) - (high - close)) / denom.replace(0, float("nan"))
        mfv = mfm * vol
        cmf = mfv.rolling(period).sum() / vol.rolling(period).sum()
        val = float(cmf.dropna().iloc[-1]) if not cmf.dropna().empty else 0.0
        return max(-1.0, min(1.0, val))
    except Exception:
        return 0.0


def cmf_to_score(cmf_val: float) -> int:
    """CMF (-1~+1) → 0-100。+0.1 → ~75, 0 → 50, -0.1 → ~25。"""
    return int(max(0, min(100, 50 + cmf_val * 250)))


# ============================================================
# TWSE T86 全市場抓取（一天一次，快取）
# ============================================================

def _recent_trade_dates(n: int = 5) -> List[str]:
    """回傳最近 n 個可能的交易日字串 YYYYMMDD（不含週末，不驗假日）。"""
    result = []
    d = date.today()
    while len(result) < n:
        if d.weekday() < 5:   # 0=Mon … 4=Fri
            result.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return result


def _fetch_t86_one_day(date_str: str) -> Optional[Dict[str, dict]]:
    """抓 TWSE T86 單日全市場三大法人買賣超（上市）。
    返回 {stock_code: {foreign_net, trust_net, dealer_net, total_net}} 或 None。
    """
    import urllib.request, json
    url = (
        f"https://www.twse.com.tw/rwd/zh/fund/T86"
        f"?date={date_str}&selectType=ALL&response=json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
        data = json.loads(raw)
    except Exception as e:
        print(f"[chip_flow] T86 {date_str} 取得失敗: {e}")
        return None

    if data.get("stat") != "OK":
        return None

    fields: List[str] = data.get("fields", [])
    rows: List[List] = data.get("data", [])
    if not fields or not rows:
        return None

    # 尋找欄位 index（以名稱比對，不寫死）
    def _find(patterns: List[str]) -> int:
        for p in patterns:
            for i, f in enumerate(fields):
                if p in f:
                    return i
        return -1

    i_code    = _find(["證券代號", "股票代號"])
    i_foreign = _find(["外陸資買賣超股數"])
    i_trust   = _find(["投信買賣超股數"])
    i_dealer  = _find(["自營商買賣超股數"])
    i_total   = _find(["三大法人買賣超股數"])

    if i_code < 0 or i_foreign < 0:
        print(f"[chip_flow] T86 fields 找不到必要欄位: {fields}")
        return None

    def _parse_num(s: str) -> int:
        try:
            return int(str(s).replace(",", "").replace("+", "").strip() or "0")
        except Exception:
            return 0

    result: Dict[str, dict] = {}
    for row in rows:
        if len(row) <= max(i_code, i_foreign):
            continue
        code = str(row[i_code]).strip()
        result[code] = {
            "foreign_net": _parse_num(row[i_foreign]),
            "trust_net":   _parse_num(row[i_trust])  if i_trust  >= 0 else 0,
            "dealer_net":  _parse_num(row[i_dealer]) if i_dealer >= 0 else 0,
            "total_net":   _parse_num(row[i_total])  if i_total  >= 0 else 0,
        }
    return result


def fetch_tw_chip_table(force_refresh: bool = False) -> Dict[str, dict]:
    """抓今日（或最近交易日）TWSE T86 全市場籌碼表，快取在記憶體。

    返回 {stock_code: {foreign_net, trust_net, dealer_net, total_net}}；
    失敗回空 dict（呼叫端拿不到資料視為 None）。
    """
    global _TW_CACHE
    if not force_refresh:
        for k in _TW_CACHE:
            return _TW_CACHE[k]   # 記憶體裡有任何一筆就直接回

    for date_str in _recent_trade_dates(5):
        tbl = _fetch_t86_one_day(date_str)
        if tbl:
            print(f"[chip_flow] T86 {date_str} 取得 {len(tbl)} 檔")
            _TW_CACHE = {date_str: tbl}
            return tbl
    print("[chip_flow] T86 所有候選日期均失敗，降級為空表")
    return {}


def _infer_volume_from_df(df) -> Optional[float]:
    """從 yfinance df 取近 20 日平均成交量（千股）。"""
    if not _PANDAS_OK or df is None:
        return None
    try:
        vol = df["Volume"].astype(float).tail(20)
        return float(vol.mean()) if not vol.empty else None
    except Exception:
        return None


# ============================================================
# 主要公開函式
# ============================================================

def chip_flow_score(
    symbol: str,
    df=None,
    tw_table: Optional[Dict[str, dict]] = None,
) -> Optional[Dict]:
    """計算個股籌碼面分數。

    Args:
        symbol: Yahoo 代號（台股 = "2330.TW", 美股 = "NVDA"）
        df:     yfinance OHLCV DataFrame（美股用 CMF；台股可為 None）
        tw_table: 預抓的 TWSE T86 全市場表（台股用）；None 時自動抓取

    Returns:
        dict  {"score": 0-100, "label": ..., "foreign_net": ..., "trust_net": ...,
               "net_trend": ..., "source": ...}
        或 None（資料不足時，呼叫端以中性 50 代入）
    """
    is_tw = symbol.upper().endswith(".TW") or symbol.upper().endswith(".TWO")

    if is_tw:
        code = symbol.split(".")[0]
        tbl = tw_table if tw_table is not None else fetch_tw_chip_table()
        if not tbl:
            return None
        row = tbl.get(code)
        if row is None:
            return None

        foreign_net = row["foreign_net"]
        trust_net   = row["trust_net"]
        total_net   = row["total_net"]

        avg_vol = _infer_volume_from_df(df)
        if avg_vol and avg_vol > 0:
            net_ratio_pct = (foreign_net + trust_net) / avg_vol * 100.0
        else:
            net_ratio_pct = 0.0

        streak = _estimate_streak(foreign_net + trust_net)

        score = blend_tw_score(net_ratio_pct, streak)
        label = chip_label(score)
        net_str = f"外資{_fmt(foreign_net)} 投信{_fmt(trust_net)}"
        trend_str = _streak_label(streak)
        return {
            "score":       score,
            "label":       label,
            "foreign_net": foreign_net,
            "trust_net":   trust_net,
            "total_net":   total_net,
            "net_trend":   trend_str,
            "net_summary": net_str,
            "source":      "TWSE T86",
        }

    else:
        # 美股：用 CMF
        if df is None or not _PANDAS_OK:
            return None
        cmf_val = compute_cmf(df)
        score = cmf_to_score(cmf_val)
        label = chip_label(score)
        return {
            "score":       score,
            "label":       label,
            "foreign_net": None,
            "trust_net":   None,
            "total_net":   None,
            "net_trend":   f"CMF={cmf_val:+.3f}",
            "net_summary": f"Chaikin Money Flow(20)={cmf_val:+.3f}",
            "source":      "CMF",
        }


def _fmt(n: int) -> str:
    """千張格式化：3456789 → +3456 張"""
    lot = n // 1000
    sign = "+" if lot >= 0 else ""
    return f"{sign}{lot:,}張"


def _estimate_streak(net_today: int) -> int:
    """單日淨買賣估算連買/賣 streak（無多日歷史時的簡化版）。
    正值估 +1（連買 1 日），負值估 -1（連賣 1 日）。
    呼叫端如有多日表可自行計算更準確的 streak 再傳入 blend_tw_score。
    """
    if net_today > 0:
        return 1
    if net_today < 0:
        return -1
    return 0


def _streak_label(streak: int) -> str:
    if streak >= 3:
        return f"連買{streak}日"
    if streak == 2:
        return "連買2日"
    if streak == 1:
        return "買超"
    if streak == 0:
        return "中性"
    if streak == -1:
        return "賣超"
    if streak == -2:
        return "連賣2日"
    return f"連賣{abs(streak)}日"
