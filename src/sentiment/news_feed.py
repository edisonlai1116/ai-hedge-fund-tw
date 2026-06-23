"""免 API Key 的新聞輿情模組。

用 Google News RSS（公開、免 Key）抓取：
- 總經 / 地緣政治新聞（川普關稅、戰爭衝突、Fed 利率…）→ 市場級情緒。
- 個股新聞標題 → 個股情緒。

情緒以中英關鍵字詞庫計分（0–100，50 為中性），不需要任何 LLM 或付費 API。
所有對外抓取都包在 try/except；失敗一律回中性、不丟例外、不臆造。

注意：關鍵字情緒是輕量啟發式，僅供「輿情參考因素之一」，非精準情緒分析。
要更準可在有 API key 時改用 LLM（見 src/agents/news_sentiment.py）。
"""
from __future__ import annotations

import urllib.parse
from typing import Dict, List

# 市場級總經 / 地緣政治查詢（使用者特別點名：川普、戰爭等會影響股市走勢的新聞）。
MACRO_QUERIES = [
    "stock market outlook",
    "Trump tariffs trade policy stock market",
    "war geopolitical conflict markets",
    "Federal Reserve interest rate decision",
    "inflation CPI economy",
]

# 多頭 / 利多關鍵字（中英混合，全部小寫比對）。
_POSITIVE = {
    "surge", "rally", "rallies", "soar", "soars", "jump", "jumps", "gain", "gains",
    "beat", "beats", "record", "high", "highs", "upgrade", "upgraded", "bullish",
    "boom", "strong", "strength", "growth", "rebound", "recovery", "optimism",
    "optimistic", "outperform", "breakthrough", "demand", "profit", "profits",
    "rate cut", "cuts rates", "easing", "ceasefire", "peace deal", "deal",
    "上漲", "大漲", "飆", "創高", "新高", "看多", "利多", "強勢", "成長", "回升",
    "反彈", "突破", "樂觀", "降息", "停火", "和談", "獲利", "超預期",
}
# 空頭 / 利空關鍵字。
_NEGATIVE = {
    "plunge", "plunges", "crash", "crashes", "slump", "tumble", "tumbles", "fall",
    "falls", "drop", "drops", "loss", "losses", "miss", "misses", "downgrade",
    "downgraded", "bearish", "selloff", "sell-off", "recession", "fear", "fears",
    "weak", "weakness", "slowdown", "warning", "warns", "cut guidance", "layoff",
    "layoffs", "sanction", "sanctions", "tariff", "tariffs", "war", "conflict",
    "invasion", "attack", "rate hike", "hikes rates", "default", "crisis",
    "下跌", "大跌", "崩", "暴跌", "重挫", "看空", "利空", "弱勢", "衰退", "恐慌",
    "下修", "示警", "警告", "裁員", "制裁", "關稅", "戰爭", "衝突", "升息", "危機",
}


def _clamp(v: float, lo: float = 0, hi: float = 100) -> int:
    return int(max(lo, min(hi, round(v))))


def keyword_sentiment(headlines: List[str]) -> Dict:
    """以關鍵字詞庫對一組標題計分。回傳 score 0–100（50 中性）、命中數與標題數。

    純函式、無網路；可離線單元測試。
    """
    headlines = [h for h in (headlines or []) if h]
    if not headlines:
        return {"score": 50, "label": "中性／無資料", "pos": 0, "neg": 0, "n": 0}

    pos = neg = 0
    for h in headlines:
        low = h.lower()
        for w in _POSITIVE:
            if w in low:
                pos += 1
        for w in _NEGATIVE:
            if w in low:
                neg += 1

    total_hits = pos + neg
    if total_hits == 0:
        score = 50
    else:
        # 命中比例決定偏離中性的幅度；命中越多、越一面倒，偏離越大。
        net = (pos - neg) / total_hits          # -1 .. 1
        score = _clamp(50 + net * 45)            # 5 .. 95
    return {"score": score, "label": _label(score), "pos": pos, "neg": neg, "n": len(headlines)}


def _label(score: int) -> str:
    if score >= 65:
        return "偏多"
    if score < 45:
        return "偏空"
    return "中性"


def _google_news_titles(query: str, limit: int = 20) -> List[str]:
    """抓 Google News RSS 標題（免 Key）。任何失敗回空清單。"""
    try:
        import feedparser  # 專案相依；CI 已安裝
    except Exception:
        return []
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        return [e.title for e in feed.entries[:limit] if getattr(e, "title", None)]
    except Exception:
        return []


def macro_sentiment() -> Dict:
    """彙整總經 / 地緣政治新聞 → 市場級情緒（score 0–100）。失敗回中性。"""
    all_titles: List[str] = []
    per_query = []
    for q in MACRO_QUERIES:
        titles = _google_news_titles(q, limit=10)
        all_titles.extend(titles)
        per_query.append({"query": q, "count": len(titles)})

    result = keyword_sentiment(all_titles)
    result["queries"] = per_query
    # 取最具代表性的幾則標題給前端顯示。
    result["sample_headlines"] = all_titles[:8]
    return result


def ticker_news_sentiment(ticker: str, company_hint: str = "") -> Dict:
    """單一個股的新聞標題情緒。失敗回中性。

    回傳包含 "titles" 欄位供 catalysts 模組複用（避免重複抓取）。
    """
    base = ticker.split(".")[0]
    query = f"{base} stock {company_hint}".strip()
    titles = _google_news_titles(query, limit=15)
    result = keyword_sentiment(titles)
    result["sample_headlines"] = titles[:5]
    result["titles"] = titles   # 供 catalysts.classify_events 複用
    return result
