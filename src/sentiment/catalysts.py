"""催化因素 / 事件風險警報分類器（免 API Key）。

吃一組已抓好的新聞標題（titles），用中英雙語關鍵字分成：
  - catalysts（利多/催化）：法說會、財報超預期、訂單、購併、庫藏股、新產品…
  - risks（風險警報）：財測下修、調查/訴訟、解禁、減產、庫存調整、降評…

純函式（classify_events 吃 titles list）→ 可離線單元測試，無 I/O。
與 news_feed 互補：news_feed 算情緒分數；本模組提供結構化事件清單。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# -------------------------------------------------------
# 催化（利多/正面事件）關鍵字
# -------------------------------------------------------
CATALYST_PATTERNS: List[Tuple[str, str]] = [
    # (tag, pattern)
    ("財報優於預期", r"beat|beats|exceed|better.than.expected|超預期|優於預期|盈餘優於|獲利超"),
    ("法說會/投資人日",  r"investor day|analyst day|earnings call|法說會|投資人日"),
    ("訂單/接單",        r"order|orders|win bid|large order|獲單|接單|拿到大單|拿下訂單|重大訂單"),
    ("新產品/量產",      r"launch|launches|new product|mass produc|ramp|量產|新品|新產品|出貨|新世代"),
    ("庫藏股/回購",      r"buyback|buy.?back|repurchas|share repurchas|庫藏股|買回"),
    ("購併/合作",        r"acqui|merger|partner|joint venture|collaborate|合作|購併|收購|策略聯盟"),
    ("調升財測/上修",    r"raise|raises|upgrad|upgrade|bullish|上修|調升|上調|正向"),
    ("法人加碼",         r"institutional buy|增持|法人買超|外資買超|投信買超"),
    ("分割/股息",        r"split|dividend|分拆|配息|除權|股票分割|特別股息"),
    ("獲利創高",         r"record profit|all.time high|record high|獲利創高|史上新高|創新高|歷史新高"),
]

# -------------------------------------------------------
# 風險警報關鍵字
# -------------------------------------------------------
RISK_PATTERNS: List[Tuple[str, str]] = [
    ("財測下修",         r"lower.?s? guidance|cut guidance|miss|misses|下修|財測下修|調降|下調"),
    ("調查/訴訟",        r"investigat|lawsuit|sue|litigation|probe|regulatory|調查|訴訟|起訴|合規|監管"),
    ("解禁/大股東減持",  r"unlock|lock.?up expir|insider sell|解禁|大股東減持|大戶出脫|持股解禁"),
    ("減產/停產",        r"cut produc|halt produc|減產|停產|縮減"),
    ("庫存調整",         r"inventory adj|channel inventory|庫存調整|去庫存|庫存回補"),
    ("出貨下滑",         r"shipment declin|demand soft|出貨下滑|出貨減少|需求疲弱"),
    ("淨利衰退",         r"profit declin|net income declin|loss|虧損|淨利衰退|獲利衰退|大幅下滑"),
    ("降評",             r"downgrad|underperform|reduce|sell rating|降評|調降評等|賣出評等"),
    ("裁員/重組",        r"layoff|lay.?off|restructur|cost.cut|裁員|組織重整|人力縮減"),
    ("法規/關稅風險",    r"tariff|sanction|ban|regulation|關稅|制裁|禁令|法規限制"),
]


def classify_events(titles: List[str]) -> Dict:
    """對一組標題分類催化/風險事件。

    Returns:
        {"catalysts": [{"tag": str, "headline": str}],
         "risks":     [{"tag": str, "headline": str}]}
    """
    catalysts: List[Dict] = []
    risks:     List[Dict] = []
    seen_cat = set()
    seen_risk = set()

    for title in (titles or []):
        low = title.lower()
        for tag, pat in CATALYST_PATTERNS:
            if re.search(pat, low, re.I) and tag not in seen_cat:
                catalysts.append({"tag": tag, "headline": title})
                seen_cat.add(tag)
        for tag, pat in RISK_PATTERNS:
            if re.search(pat, low, re.I) and tag not in seen_risk:
                risks.append({"tag": tag, "headline": title})
                seen_risk.add(tag)

    return {"catalysts": catalysts, "risks": risks}


def has_risk(events: Dict) -> bool:
    """是否有任何風險警報（供前端顯示 ⚠ 徽章）。"""
    return bool(events and events.get("risks"))
