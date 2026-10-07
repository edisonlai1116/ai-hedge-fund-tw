"""市場情緒：VIX、恐懼貪婪指數（CNN 即時 + 可回測的自建版本）、各類股情緒。

為什麼要自建恐懼貪婪：CNN 的 Fear & Greed 只提供近一年歷史，無法回測；這裡用同樣的成分、只用當日以前的資料重建
（point-in-time），跟 CNN 近一年的值相關性見 validate_proxy()。網站顯示 CNN 即時值（抓不到時改顯示自建值），
規則判斷一律用自建值，讓回測與每日建議用同一把尺。

成分（各自換算為「近 252 日百分位」0~100 後平均，越高越貪婪）：
  1. 市場動能   SPY / 125 日均線
  2. 股價強度   11 個類股 ETF 距 52 週高點的中位數
  3. 市場廣度   11 個類股 ETF 站上 50 日線的平均幅度
  4. 波動度     VIX / VIX 50 日均（反向）
  5. 期限結構   VIX3M / VIX（< 1 = 逆價差 = 恐慌；取代 CNN 的 put/call，Yahoo 無歷史）
  6. 避險需求   SPY 20 日報酬 − TLT 20 日報酬
  7. 垃圾債需求 HYG 20 日報酬 − IEF 20 日報酬
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通訊", "XLY": "非必需消費", "XLF": "金融", "XLV": "醫療", "XLI": "工業",
    "XLE": "能源", "XLU": "公用/電力", "XLP": "必需消費", "XLB": "原物料", "XLRE": "房地產",
}
THEME_ETFS = {"SMH": "半導體", "IGV": "軟體"}
SENTIMENT_TICKERS = ["^VIX", "^VIX3M", "SPY", "QQQ", "TLT", "IEF", "HYG", "^TWII"] + list(SECTOR_ETFS) + list(THEME_ETFS)

# 情緒分級（自建恐懼貪婪 0~100）
EXTREME_FEAR, FEAR, GREED, EXTREME_GREED = 25, 45, 55, 75
VIX_PANIC = 30.0


def _naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    i = pd.to_datetime(s.index)
    s.index = i.tz_localize(None) if i.tz is not None else i
    return s.sort_index()


def _pctrank(s: pd.Series, window: int = 252) -> pd.Series:
    """每日數值在「過去 window 日（含當日）」的百分位，0~100；只用當日以前的資料。"""
    return s.rolling(window, min_periods=120).apply(lambda a: (a[:-1] < a[-1]).mean() * 100 + (a[:-1] == a[-1]).mean() * 50,
                                                    raw=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def fear_greed_components(c: Dict[str, pd.Series]) -> pd.DataFrame:
    """c: {ticker: close}。回傳各成分原始值（未標準化）。"""
    c = {k: _naive(v.dropna()) for k, v in c.items() if v is not None and len(v.dropna())}
    spy = c["SPY"]
    idx = spy.index
    get = lambda k: c[k].reindex(idx).ffill(limit=5) if k in c else None
    comp = pd.DataFrame(index=idx)
    comp["momentum"] = spy / spy.rolling(125).mean() - 1
    sect = pd.DataFrame({k: get(k) for k in SECTOR_ETFS if k in c})
    if not sect.empty:
        comp["strength"] = (sect / sect.rolling(252, min_periods=120).max() - 1).median(axis=1)
        comp["breadth"] = (sect / sect.rolling(50).mean() - 1).mean(axis=1)
    vix = get("^VIX")
    if vix is not None:
        comp["volatility"] = -(vix / vix.rolling(50).mean() - 1)
        v3 = get("^VIX3M")
        if v3 is not None:
            comp["term_structure"] = v3 / vix - 1
    for name, a, b in (("safe_haven", "SPY", "TLT"), ("junk_bond", "HYG", "IEF")):
        if a in c and b in c:
            comp[name] = get(a).pct_change(20) - get(b).pct_change(20)
    return comp


def fear_greed_proxy(c: Dict[str, pd.Series]) -> pd.DataFrame:
    """回傳 DataFrame：各成分百分位 + fg（平均，0~100）。"""
    comp = fear_greed_components(c)
    ranked = comp.apply(_pctrank)
    ranked["fg"] = ranked.mean(axis=1, skipna=True).where(ranked.notna().sum(axis=1) >= 4)
    return ranked


def fg_label(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v < EXTREME_FEAR:
        return "極度恐懼"
    if v < FEAR:
        return "恐懼"
    if v <= GREED:
        return "中性"
    if v <= EXTREME_GREED:
        return "貪婪"
    return "極度貪婪"


def fetch_cnn_fear_greed() -> Optional[Dict]:
    """CNN Fear & Greed 即時值（需瀏覽器標頭，否則 418）。失敗回 None。"""
    try:
        import requests
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
                     "Accept": "application/json, text/plain, */*", "Origin": "https://edition.cnn.com",
                     "Referer": "https://edition.cnn.com/markets/fear-and-greed"},
            timeout=20,
        )
        r.raise_for_status()
        j = r.json()
        fg = j["fear_and_greed"]
        hist = [(pd.Timestamp(int(p["x"]), unit="ms").normalize(), float(p["y"])) for p in j.get("fear_and_greed_historical", {}).get("data", [])]
        return {
            "score": round(float(fg["score"]), 1), "rating": fg.get("rating"), "timestamp": fg.get("timestamp"),
            "previous_close": _r1(fg.get("previous_close")), "previous_1_week": _r1(fg.get("previous_1_week")),
            "previous_1_month": _r1(fg.get("previous_1_month")), "history": hist,
        }
    except Exception as e:
        print(f"[sentiment] CNN Fear & Greed 抓取失敗：{type(e).__name__}: {e}")
        return None


def _r1(v):
    return round(float(v), 1) if v is not None else None


def validate_proxy(proxy_fg: pd.Series, cnn: Optional[Dict]) -> Optional[Dict]:
    """自建 vs CNN 近一年的相關性（同日、水準值）。"""
    if not cnn or not cnn.get("history"):
        return None
    s = pd.Series({d: v for d, v in cnn["history"]}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    j = pd.concat([proxy_fg.rename("proxy"), s.rename("cnn")], axis=1).dropna()
    if len(j) < 60:
        return None
    return {"n_days": int(len(j)), "corr": round(float(j["proxy"].corr(j["cnn"])), 2),
            "mean_abs_diff": round(float((j["proxy"] - j["cnn"]).abs().mean()), 1)}


def sector_sentiment(c: Dict[str, pd.Series], groups: Optional[Dict[str, List[str]]] = None,
                     universe_close: Optional[pd.DataFrame] = None) -> List[Dict]:
    """各類股情緒：類股 ETF（美股 11 類 + 半導體/軟體）與自訂族群（股票池內等權）。

    每一列：1/3/6 個月報酬、相對 SPY 3 個月、RSI14、距 50 日線、站上 50 日線比例（族群）、情緒標籤。
    標籤：過熱（RSI ≥ 70 且距 50 日線 ≥ 8%）、強勢、中性、弱勢、超賣（RSI ≤ 30 或距 50 日線 ≤ -10%）。
    """
    rows = []
    spy = _naive(c["SPY"].dropna()) if "SPY" in c else None
    spy3 = float(spy.iloc[-1] / spy.iloc[-64] - 1) if spy is not None and len(spy) > 64 else None

    def one(name, label, s, breadth=None, kind="etf"):
        s = s.dropna()
        if len(s) < 130:
            return None
        r = lambda n: float(s.iloc[-1] / s.iloc[-n - 1] - 1)
        ma50 = float(s.tail(50).mean())
        rs = float(rsi(s).iloc[-1])
        dist = float(s.iloc[-1] / ma50 - 1)
        r3 = r(63)
        rel = r3 - spy3 if spy3 is not None else None
        if rs >= 70 and dist >= 0.08:
            mood = "過熱"
        elif rs <= 30 or dist <= -0.10:
            mood = "超賣"
        elif rel is not None and rel > 0.03 and dist > 0:
            mood = "強勢"
        elif rel is not None and rel < -0.03 and dist < 0:
            mood = "弱勢"
        else:
            mood = "中性"
        return {"key": name, "name": label, "kind": kind, "ret_1m_pct": round(r(21) * 100, 1), "ret_3m_pct": round(r3 * 100, 1),
                "ret_6m_pct": round(r(126) * 100, 1), "rel_spy_3m_pct": round(rel * 100, 1) if rel is not None else None,
                "rsi14": round(rs, 0), "dist_ma50_pct": round(dist * 100, 1),
                "breadth_above_ma50_pct": round(breadth * 100, 0) if breadth is not None else None, "mood": mood}

    for k, lab in {**SECTOR_ETFS, **THEME_ETFS}.items():
        if k in c:
            row = one(k, lab, _naive(c[k]))
            if row:
                rows.append(row)
    if groups and universe_close is not None:
        u = universe_close.copy()
        u.index = pd.to_datetime(u.index)
        if u.index.tz is not None:
            u.index = u.index.tz_localize(None)
        for g, syms in groups.items():
            cols = [s for s in syms if s in u.columns]
            if len(cols) < 2:
                continue
            sub = u[cols].dropna(how="all").tail(300).ffill(limit=3)
            idxs = (1 + sub.pct_change().mean(axis=1).fillna(0)).cumprod()
            br = float((sub.iloc[-1] > sub.rolling(50).mean().iloc[-1]).mean())
            row = one(g, g, idxs, breadth=br, kind="group")
            if row:
                row["members"] = len(cols)
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 每日快照（strategy_report 呼叫；結果寫進 strategy.json 的 sentiment 欄位）
# ---------------------------------------------------------------------------
COMPONENT_LABELS = {
    "momentum": "市場動能（SPY vs 125 日線）", "strength": "股價強度（類股距 52 週高）", "breadth": "市場廣度（類股 vs 50 日線）",
    "volatility": "波動度（VIX vs 50 日均）", "term_structure": "VIX 期限結構（3 個月 / 1 個月）",
    "safe_haven": "避險需求（股 vs 長債 20 日）", "junk_bond": "垃圾債需求（HYG vs IEF 20 日）",
}


def _last(s: Optional[pd.Series]) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def market_sentiment(groups: Optional[Dict[str, List[str]]] = None, universe_close: Optional[pd.DataFrame] = None,
                     closes: Optional[Dict[str, pd.Series]] = None) -> Dict:
    """今日市場情緒快照：自建恐懼貪婪（規則用）、CNN（顯示用）、VIX、各類股情緒。"""
    if closes is None:
        from src.strategy.momentum import download_closes
        pm = download_closes(SENTIMENT_TICKERS, period="3y")
        closes = {k: f["Close"] for k, f in pm.items()}
    fg = fear_greed_proxy(closes)
    fg_s = fg["fg"].dropna()
    today = float(fg_s.iloc[-1]) if len(fg_s) else None
    vix = _naive(closes["^VIX"].dropna()) if "^VIX" in closes else None
    v3 = _naive(closes["^VIX3M"].dropna()) if "^VIX3M" in closes else None
    vix_now = _last(vix)
    v3_now = _last(v3)
    cnn = fetch_cnn_fear_greed()
    val = validate_proxy(fg["fg"], cnn)
    if cnn:
        cnn = {k: v for k, v in cnn.items() if k != "history"}

    def ago(n):
        return round(float(fg_s.iloc[-1 - n]), 1) if len(fg_s) > n else None

    twii = _naive(closes["^TWII"].dropna()) if "^TWII" in closes else None
    tw = None
    if twii is not None and len(twii) > 200:
        tw = {"close": round(float(twii.iloc[-1]), 0), "vs_ma200_pct": round((float(twii.iloc[-1]) / float(twii.tail(200).mean()) - 1) * 100, 1),
              "rsi14": round(float(rsi(twii).iloc[-1]), 0), "ret_1m_pct": round((float(twii.iloc[-1] / twii.iloc[-22]) - 1) * 100, 1),
              "dd_52w_pct": round((float(twii.iloc[-1]) / float(twii.tail(252).max()) - 1) * 100, 1)}
    asof = fg_s.index[-1].strftime("%Y-%m-%d") if len(fg_s) else None
    return {
        "asof": asof,
        "fear_greed": {
            "score": round(today, 1) if today is not None else None, "label": fg_label(today),
            "prev_1d": ago(1), "prev_1w": ago(5), "prev_1m": ago(21),
            "components": [{"key": k, "name": COMPONENT_LABELS.get(k, k), "score": round(float(fg[k].dropna().iloc[-1]), 0)}
                           for k in fg.columns if k != "fg" and len(fg[k].dropna())],
            "history": [{"date": d.strftime("%Y-%m-%d"), "fg": round(float(v), 1)} for d, v in fg_s.tail(260).items()],
            "note": "自建版本（只用當日以前資料，可回測），規則判斷用這個值；與 CNN 近一年相關性見 cnn_validation。",
        },
        "cnn": cnn,
        "cnn_validation": val,
        "vix": {"value": round(vix_now, 2) if vix_now else None,
                "prev_1d": round(float(vix.iloc[-2]), 2) if vix is not None and len(vix) > 1 else None,
                "ma50": round(float(vix.tail(50).mean()), 2) if vix is not None and len(vix) >= 50 else None,
                "vix3m": round(v3_now, 2) if v3_now else None,
                "backwardation": bool(vix_now and v3_now and vix_now > v3_now),
                "panic": bool(vix_now and vix_now >= VIX_PANIC)},
        "taiex": tw,
        "sectors": sector_sentiment(closes, groups, universe_close),
    }
