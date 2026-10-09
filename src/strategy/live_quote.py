"""即時報價（盤中「今日漲跌」用）。

策略排名 strategy.json 由 GitHub Actions 每天 06:30 / 17:00（台北）產生，盤中看時台股還停在前一日收盤，
「今日漲跌」其實是昨天的漲跌。這裡在請求當下抓即時價，覆蓋持股的現價與漲跌：
- 台股：TWSE 基本市況 API（mis.twse.com.tw，上市 tse_ / 上櫃 otc_，一次批次查），z=成交價、y=昨收。
- 美股：yfinance fast_info（last_price / previous_close）。
抓不到就回空，呼叫端沿用每日快照，不讓請求失敗。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional

_CACHE: Dict[str, Dict] = {}
_TTL = 30.0
_MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"


def _num(v) -> Optional[float]:
    try:
        x = float(str(v).split("_")[0])
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def _tw_quotes(symbols: Iterable[str]) -> Dict[str, Dict]:
    import requests

    syms = list(symbols)
    if not syms:
        return {}
    chans = []
    for s in syms:
        code, _, sfx = s.partition(".")
        chans.append(f"{'otc' if sfx == 'TWO' else 'tse'}_{code}.tw")
    out: Dict[str, Dict] = {}
    for i in range(0, len(chans), 40):
        r = requests.get(_MIS_URL, params={"ex_ch": "|".join(chans[i:i + 40]), "json": "1", "delay": "0"},
                         headers={"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"},
                         timeout=8)
        for x in (r.json() or {}).get("msgArray", []):
            code, ex = x.get("c"), x.get("ex")
            prev = _num(x.get("y"))
            # z 在該撮合區間沒成交時是 "-"：退用最佳買價，再退用開盤價
            price = _num(x.get("z")) or _num(x.get("b")) or _num(x.get("o"))
            if not code or not prev or not price:
                continue
            d, t = x.get("d", ""), x.get("t", "")
            out[f"{code}.{'TWO' if ex == 'otc' else 'TW'}"] = {
                "price": price, "prev_close": prev,
                "change_pct": round((price / prev - 1) * 100, 2),
                "as_of": f"{d[:4]}-{d[4:6]}-{d[6:]} {t}" if len(d) == 8 else None,
                "source": "twse",
            }
    return out


def _us_quote(sym: str) -> Optional[Dict]:
    import yfinance as yf

    fi = yf.Ticker(sym).fast_info
    price, prev = _num(fi.get("lastPrice")), _num(fi.get("previousClose"))
    if not price or not prev:
        return None
    return {"price": round(price, 4), "prev_close": prev, "change_pct": round((price / prev - 1) * 100, 2),
            "as_of": None, "source": "yfinance"}


def live_quotes(symbols: Iterable[str]) -> Dict[str, Dict]:
    """symbols 為 yfinance 格式（2330.TW / 6488.TWO / NVDA）。回傳 {symbol: {price, prev_close, change_pct, as_of, source}}。"""
    now = time.time()
    syms = list(dict.fromkeys(symbols))
    out = {s: _CACHE[s] for s in syms if s in _CACHE and now - _CACHE[s]["ts"] < _TTL}
    need = [s for s in syms if s not in out]
    tw = [s for s in need if s.endswith((".TW", ".TWO"))]
    us = [s for s in need if s not in tw]
    fresh: Dict[str, Dict] = {}
    try:
        fresh.update(_tw_quotes(tw))
    except Exception:
        pass
    # 台股：代號後綴猜錯（上市/上櫃）時換另一個再查一次
    miss = [s for s in tw if s not in fresh]
    if miss:
        alt = {(s[:-4] + ".TW" if s.endswith(".TWO") else s[:-3] + ".TWO"): s for s in miss}
        try:
            for k, q in _tw_quotes(alt).items():
                fresh[alt[k]] = q
        except Exception:
            pass
    if us:
        def one(s):
            try:
                return s, _us_quote(s)
            except Exception:
                return s, None
        with ThreadPoolExecutor(max_workers=min(8, len(us))) as ex:
            for s, q in ex.map(one, us):
                if q:
                    fresh[s] = q
    for s, q in fresh.items():
        q["ts"] = now
        _CACHE[s] = q
    out.update(fresh)
    return out
