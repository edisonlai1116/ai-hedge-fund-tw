"""近 30 天新聞／事件 → catalyst_score（即時；歷史新聞無 point-in-time 來源，不進回測）。

每個事件評估：
  direction  +1 / 0 / -1
  magnitude  0~100（事件類別的典型影響力 × 標題強度）
  duration   short / medium / long
  priced_in  0~100：事件後股價相對大盤已漲了多少（漲越多代表越被反映）
catalyst_score = 50 + Σ direction × magnitude × (1 − priced_in) × 時間衰減，夾在 0~100。

重點是區分「利多」與「市場已經漲很多的利多」：同樣的大單新聞，若事件後已經跑贏大盤 25%，
剩下的影響力只算原本的一小部分。

分類採關鍵字規則（確定性）；LLM 若可用只負責補充解讀文字，不改分數。
"""
from __future__ import annotations

import math
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

# 類別：(關鍵字 regex, 典型 magnitude, duration)
CATEGORIES = {
    "earnings": (r"\b(earnings|quarter(ly)? results|q[1-4] results|beats?|miss(es|ed)?|eps)\b", 70, "medium"),
    "guidance": (r"\b(guidance|outlook|forecast|raises? (its )?(full-year|annual)|cuts? (its )?(forecast|outlook))\b", 80, "long"),
    "major_customer": (r"\b(customer|order|wins? (a )?deal|design win|supply agreement)\b", 55, "medium"),
    "hyperscaler_contract": (r"\b(microsoft|google|alphabet|amazon|aws|meta|oracle|openai|xai|anthropic)\b.*\b(deal|contract|agreement|partnership|ppa|order)\b|\b(deal|contract|agreement|partnership|ppa)\b.*\b(microsoft|google|amazon|aws|meta|oracle|openai)\b", 80, "long"),
    "government_contract": (r"\b(government|pentagon|department of|doe|dod|federal|chips act|grant|subsid)", 60, "long"),
    "product_launch": (r"\b(launch(es|ed)?|unveil|introduc|new (chip|product|platform)|sampling|ships?)\b", 40, "medium"),
    "capacity_expansion": (r"\b(expan(d|sion)|new (fab|plant|facility)|capacity|capex|build(s|ing)? (a )?(plant|data center))\b", 45, "long"),
    "m_and_a": (r"\b(acquir|acquisition|merger|merge|buyout|takeover|to buy)\b", 70, "long"),
    "analyst_revision": (r"\b(upgrade|downgrade|price target|initiat|overweight|underweight|buy rating|sell rating)\b", 35, "short"),
    "regulatory": (r"\b(regulat|antitrust|ftc|sec probe|investigation|export (control|ban|restriction)|tariff|sanction|nrc)\b", 65, "long"),
    "competitor_action": (r"\b(rival|competitor|competition|price war|loses? share)\b", 40, "medium"),
    "supply_increase": (r"\b(oversupply|glut|inventory build|excess inventory|supply increase)\b", 55, "medium"),
    "demand_increase": (r"\b(demand (surge|boom|strong|soar)|shortage|sold out|backlog|record demand|ai demand)\b", 55, "medium"),
}
POSITIVE = r"\b(beat|beats|raise|raises|raised|record|surge|soar|jump|win|wins|upgrade|strong|boost|approv|expand|partnership|deal|outperform|accelerat|above)\b"
NEGATIVE = r"\b(miss|misses|missed|cut|cuts|lower|lowers|plunge|tumble|slump|fall|falls|drop|downgrade|weak|probe|lawsuit|ban|delay|halt|below|warn|warning|recall|decline)\b"
DURATION_HALF_LIFE = {"short": 5, "medium": 15, "long": 45}   # 天


def fetch_news(query: str, days: int = 30, limit: int = 40) -> List[Dict]:
    """Google News RSS（免 Key）：回傳 [{title, published(datetime UTC), source}]。"""
    try:
        import feedparser
    except Exception:
        return []
    q = urllib.parse.quote(f"{query} when:{days}d")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    out = []
    for e in feed.entries[:limit]:
        try:
            pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue
        out.append({"title": e.get("title", ""), "published": pub, "source": (e.get("source") or {}).get("title", "")})
    return out


def classify(title: str) -> Dict:
    t = title.lower()
    cats = [c for c, (rx, _, _) in CATEGORIES.items() if re.search(rx, t)]
    pos, neg = len(re.findall(POSITIVE, t)), len(re.findall(NEGATIVE, t))
    direction = 1 if pos > neg else (-1 if neg > pos else 0)
    if not cats:
        return {"categories": [], "direction": direction, "magnitude": 0, "duration": "short"}
    main = max(cats, key=lambda c: CATEGORIES[c][1])
    mag = CATEGORIES[main][1] * (1.0 if abs(pos - neg) >= 1 else 0.6)
    return {"categories": cats, "category": main, "direction": direction, "magnitude": round(mag), "duration": CATEGORIES[main][2]}


def priced_in(close: Optional[pd.Series], bench: Optional[pd.Series], when: datetime, direction: int) -> Optional[float]:
    """事件後股價相對大盤的超額報酬，換算成「已反映」程度 0~100（方向一致才算反映）。"""
    if close is None or bench is None or direction == 0:
        return None
    c, b = _naive(close), _naive(bench)
    d = pd.Timestamp(when).tz_localize(None).normalize()
    c_after, b_after = c[c.index >= d], b[b.index >= d]
    c_before, b_before = c[c.index < d], b[b.index < d]
    if c_after.empty or c_before.empty or b_after.empty or b_before.empty:
        return None
    excess = (c_after.iloc[-1] / c_before.iloc[-1] - 1) - (b_after.iloc[-1] / b_before.iloc[-1] - 1)
    moved = excess * direction           # 往事件方向走了多少
    return float(max(0.0, min(100.0, moved / 0.25 * 100)))   # 已超額 25% 視為完全反映


def catalyst_report(symbol: str, company: str = "", close: Optional[pd.Series] = None,
                    bench: Optional[pd.Series] = None, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.now(timezone.utc)
    q = f"{symbol} stock" if not company else f"\"{company}\" OR {symbol} stock"
    items = [it for it in fetch_news(q) if _relevant(it["title"], symbol, company)]
    events = []
    for it in items:
        c = classify(it["title"])
        if not c["categories"]:
            continue
        age = max((now - it["published"]).days, 0)
        pin = priced_in(close, bench, it["published"], c["direction"])
        decay = 0.5 ** (age / DURATION_HALF_LIFE[c["duration"]])
        effective = c["direction"] * c["magnitude"] * (1 - (pin or 0) / 100) * decay
        events.append({**c, "title": it["title"], "published": it["published"].date().isoformat(), "age_days": age,
                       "priced_in": None if pin is None else round(pin), "effective": round(effective, 1)})
    events.sort(key=lambda e: abs(e["effective"]), reverse=True)
    if not events:
        return {"score": None, "events": [], "news_count": len(items), "note": "近 30 天無可分類事件（不當中性）"}
    # 同類事件不重複加總：每類取影響最大的一則
    seen, total = set(), 0.0
    for e in events:
        if e["category"] in seen:
            continue
        seen.add(e["category"])
        total += e["effective"]
    score = 50 + 50 * math.tanh(total / 120)
    top = events[0]
    return {
        "score": round(score, 1),
        "events": events[:8],
        "news_count": len(items),
        "key_catalyst": f"{top['category']}（{'+' if top['direction'] > 0 else ('-' if top['direction'] < 0 else '0')}，"
                        f"已反映 {top['priced_in'] if top['priced_in'] is not None else '?'}%）：{top['title']}",
        "method": "keyword rules（確定性）；LLM 只做解讀不改分數",
    }


_SUFFIXES = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "holdings", "group", "ltd", "plc", "the",
             "company", "technologies", "technology", "energy", "systems", "class", "a", "n.v.", "sa"}


def _relevant(title: str, symbol: str, company: str) -> bool:
    """標題需提到代號（獨立字）或公司名的主要字（例：Bloom、Vistra），排除只是撞到常見字的新聞。"""
    t = title or ""
    if re.search(rf"(?<![A-Za-z]){re.escape(symbol.upper())}(?![A-Za-z])", t):
        return True
    words = [w for w in re.split(r"[\s,]+", (company or "").lower()) if w and w not in _SUFFIXES and len(w) >= 3]
    return any(re.search(rf"\b{re.escape(w)}\b", t.lower()) for w in words[:2])


def _naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    i = pd.to_datetime(s.index)
    s.index = i.tz_localize(None) if i.tz is not None else i
    return s
