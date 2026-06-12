"""免 API Key 取得「美股當日熱門榜」標的，反映每天市場的變化。

來源：Yahoo Finance 預設篩選器（day_gainers 漲幅榜、most_actives 成交量最大、day_losers 跌幅榜）。
優先用 yfinance 的 screen()，失敗再退回 Yahoo 公開 screener REST 端點；全失敗回空清單（不臆造）。

回傳的是「代號清單」，由每日 pipeline 再跑買進分數排序。
"""
from __future__ import annotations

from typing import List

# Yahoo 預設篩選器 id（公開、免 Key）
_SCREENS = ["day_gainers", "most_actives"]


def _via_yfinance(scr_id: str, count: int) -> List[str]:
    try:
        import yfinance as yf
        res = yf.screen(scr_id, count=count)  # 0.2.5x+ 支援預設篩選器字串
    except Exception:
        return []
    quotes = (res or {}).get("quotes", []) if isinstance(res, dict) else []
    return [q.get("symbol") for q in quotes if isinstance(q, dict) and q.get("symbol")]


def _via_rest(scr_id: str, count: int) -> List[str]:
    try:
        import requests
        url = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
               f"?count={count}&scrIds={scr_id.upper()}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = data["finance"]["result"][0]["quotes"]
        return [q.get("symbol") for q in rows if q.get("symbol")]
    except Exception:
        return []


def fetch_market_movers(limit: int = 40, per_screen: int = 30) -> List[str]:
    """彙整當日漲幅榜 + 成交量最大榜，去重後回傳代號清單（最多 limit 檔）。失敗回 []。"""
    seen = set()
    out: List[str] = []
    for scr in _SCREENS:
        syms = _via_yfinance(scr, per_screen) or _via_rest(scr, per_screen)
        for s in syms:
            su = str(s).upper().strip()
            # 過濾明顯非個股（指數 ^、貨幣對含 =）
            if not su or su.startswith("^") or "=" in su:
                continue
            if su not in seen:
                seen.add(su)
                out.append(su)
            if len(out) >= limit:
                return out
    return out
