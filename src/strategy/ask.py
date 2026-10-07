"""AI 問答：把「使用者問題 + 持股 + 系統今日建議 + 市場情緒 + 策略規則」組成一份提示詞。

兩種用法：
  1. 直接呼叫免費 Google Gemini（需環境變數 GOOGLE_API_KEY；忙碌時依序改用備援模型）。
  2. 不呼叫 LLM，只回傳提示詞，讓使用者複製到 ChatGPT / Claude / Gemini 網頁版詢問。

數字一律由系統確定性計算（evaluate_holdings、每日建議、換現金順序），LLM 只負責解讀與回答，
提示詞要求它不得捏造價格或改寫系統數字。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

GEMINI_MODELS = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest",
                             "gemini-flash-lite-latest", "gemini-3.5-flash-lite") if m]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LLM_TIME_BUDGET = 80          # 秒（Render 免費機單一請求約 100 秒上限）

PRESET_QUESTIONS = [
    "我現在滿倉，如果想賣出一些換現金，應該先賣哪幾檔？各賣多少？",
    "今天我的持股有哪些需要動作？為什麼？",
    "我的持股是不是太集中在某個主題？要怎麼調整？",
    "如果明天大跌 10%，我的持股哪些最危險？",
    "有新資金的話，現在該買什麼、還是先等？",
]

# 換現金順序（系統規則，未經回測；排在越前面越先賣）
_CASH_ORDER = {"賣出換股": 0, "減碼": 1}


def _fmt(v, d=0, pct=False, sign=False):
    if v is None:
        return "—"
    s = f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"
    return s + ("%" if pct else "")


def raise_cash_plan(ev: Dict, report: Dict) -> List[Dict]:
    """若要賣股換現金，建議的先後順序（確定性規則，非回測結果）：
    1. 系統已建議「賣出換股」 2. 集中度過高「減碼」到 20% 以下 3. 排名 Tier D／高風險低檔（跌深但不建議買）
    4. Tier C 5. 其他個股依動能排名由弱到強 6. 核心 ETF 7. 最後才動：低檔可買（Tier A/B 長線贏家回撤）與動能前段班。"""
    from src.strategy.momentum import CONCENTRATION_PCT, TOP_N, low_view, market_of
    out = []
    total = ev.get("total_twd") or 0
    rows_by = {r["symbol"]: r for m in report.get("markets", {}).values() for r in m.get("rows", [])}
    for h in ev.get("holdings", []):
        sym, act = h["symbol"], h["action"]
        row = rows_by.get(sym) or {}
        lv = low_view(row, report) if row else None
        rec = (lv or {}).get("recommendation")
        tier = row.get("quality_tier")
        rank = h.get("rank")
        if act in _CASH_ORDER:
            group, why = _CASH_ORDER[act], f"系統建議「{act}」"
        elif rec in ("BUY", "BUY_STAGED"):
            group, why = 7, f"低檔可買（{lv['label']}）：最後才賣"
        elif rank is not None and rank <= TOP_N.get(market_of(sym), 10) and act != "核心 ETF":
            group, why = 7, f"動能前段（第 {rank} 名）：最後才賣"
        elif rec in ("SPECULATIVE_WATCH", "NO_BUY") or tier == "D":
            group, why = 2, f"排名偏低（Tier {tier or '—'}）" + (f"、{lv['label']}" if lv else "")
        elif tier == "C":
            group, why = 3, "排名偏後（Tier C）"
        elif act == "核心 ETF":
            group, why = 5, "核心 ETF：流動性好，但屬核心部位"
        else:
            group, why = 4, f"一般持股（排名 {rank or '—'}）"
        trim = None
        if act == "減碼" and total:
            trim = max(0.0, h["value_twd"] - total * CONCENTRATION_PCT / 100)
        out.append({"symbol": sym, "group": group, "why": why, "value_twd": h["value_twd"], "rank": rank,
                    "pnl_pct": h.get("pnl_pct"), "suggest_trim_twd": round(trim) if trim else None})
    out.sort(key=lambda x: (x["group"], -(x["rank"] or 0)))
    for i, x in enumerate(out, 1):
        x["order"] = i
    return out


def build_context(holdings: List[Dict], report: Dict, advice: Optional[Dict], fx: float) -> Dict:
    from src.strategy.momentum import evaluate_holdings
    ev = evaluate_holdings(holdings, report, fx) if holdings else None
    return {"evaluation": ev, "cash_plan": raise_cash_plan(ev, report) if ev else [], "advice": advice,
            "sentiment": report.get("sentiment") or {}, "strategy": report.get("strategy") or {}}


def build_prompt(question: str, ctx: Dict) -> str:
    ev, adv, sent = ctx.get("evaluation"), ctx.get("advice") or {}, ctx.get("sentiment") or {}
    st = ctx.get("strategy") or {}
    fg, vix = (sent.get("fear_greed") or {}), (sent.get("vix") or {})
    L: List[str] = []
    L.append("你是我的投資助理。請根據下面「我的持股」與「系統今日建議」回答我的問題。")
    L.append("規則：1) 只能使用下面提供的數字，不要自行捏造股價或財報；2) 系統的動作建議是確定性規則算出的，"
             "若你不同意請說明理由，但不要假裝系統說了別的；3) 回答要具體（哪幾檔、各大約多少金額、先後順序、理由）；"
             "4) 用繁體中文，先給結論再給理由；5) 提醒這不是投資建議。")
    L.append("")
    L.append(f"## 我的問題\n{question.strip()}")
    L.append("")
    L.append(f"## 市場狀況（{adv.get('date') or sent.get('asof') or ''}）")
    L.append(f"- 恐懼貪婪指數 {_fmt(fg.get('score'))}（{fg.get('label') or '—'}）；VIX {_fmt(vix.get('value'), 2)}"
             + ("（期限結構逆價差，短線恐慌）" if vix.get("backwardation") else ""))
    if adv.get("headline"):
        L.append(f"- 系統今日結論：{adv['headline']}")
    for g in adv.get("sentiment_guidance") or []:
        L.append(f"- {g}")
    L.append("")
    if ev:
        L.append("## 我的持股（系統評估，金額為新台幣）")
        L.append(f"總資產 NT${_fmt(ev.get('total_twd'))}；今日損益 NT${_fmt(ev.get('day_pnl_twd'), sign=True)}"
                 f"（{_fmt(ev.get('day_change_pct'), 2, True, True)}）；美元匯率 {ev.get('fx_usd_twd')}")
        L.append("| 代號 | 股數 | 成本 | 現價 | 市值 NT$ | 佔比 | 總損益 | 今日 | 排名 | 系統動作 | 今天該做 | 理由 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in ev["holdings"]:
            L.append(f"| {h['symbol']} | {_fmt(h['shares'])} | {h['cost']} | {h.get('price') or '—'} | {_fmt(h['value_twd'])} | "
                     f"{h['weight_pct']}% | {_fmt(h.get('pnl_pct'), 1, True, True)} | {_fmt(h.get('day_change_pct'), 2, True, True)} | "
                     f"{h.get('rank') or '—'} | {h['action']} | {h.get('today') or '—'} | {(h.get('reason') or '').replace('|', '/')[:160]} |")
        L.append("")
        plan = ctx.get("cash_plan") or []
        if plan:
            L.append("## 系統試算：若要賣股換現金的先後順序（確定性規則，未經回測）")
            L.append("順序原則：系統已建議賣出 → 集中度過高減碼 → 排名偏低／高風險低檔 → 排名偏後 → 一般持股（動能弱的先）"
                     " → 核心 ETF → 最後才動低檔可買與動能前段班。")
            for x in plan:
                trim = f"，建議減碼約 NT${_fmt(x['suggest_trim_twd'])}" if x.get("suggest_trim_twd") else ""
                L.append(f"{x['order']}. {x['symbol']}（市值 NT${_fmt(x['value_twd'])}、損益 {_fmt(x.get('pnl_pct'), 1, True, True)}）：{x['why']}{trim}")
            L.append("")
        nb = ev.get("low_entry_buys") or {}
        if any(nb.values()):
            L.append("## 系統目前認為可買的低檔股（我還沒有的）")
            for m, items in nb.items():
                for b in items[:5]:
                    L.append(f"- {b['symbol']} {b.get('name') or ''}：{b.get('low_label') or ''}，距高點 {_fmt(b.get('dd_52w_pct'))}%")
            L.append("")
    else:
        L.append("## 我的持股\n（尚未提供持股）\n")
    for m, lab in (("us", "美股"), ("tw", "台股")):
        d = (adv.get("markets") or {}).get(m)
        if not d:
            continue
        L.append(f"## 系統今日{lab}建議：{d.get('summary')}")
        for a in (d.get("actions") or [])[:8]:
            L.append(f"- {a['action']} {a['symbol']}：{(a.get('reason') or '')[:140]}")
        for a in (d.get("watch") or [])[:6]:
            L.append(f"- 觀察 {a['symbol']}：{a['action']}")
        L.append("")
    L.append("## 策略規則摘要")
    if st.get("rule"):
        L.append(f"- {st['rule']}")
    if st.get("low_entry_rule"):
        L.append(f"- {st['low_entry_rule']}")
    L.append("- 回測有倖存者偏差（股票池是今天挑的 AI 贏家），實際報酬會較低。")
    return "\n".join(L)


def call_gemini(prompt: str, api_key: Optional[str] = None) -> Dict:
    """呼叫免費 Gemini；主模型忙碌（429/503/逾時）時依序改用備援模型。回傳 {answer, model} 或 {error}。"""
    import time
    import requests
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"error": "伺服器未設定 GOOGLE_API_KEY（免費金鑰可在 Google AI Studio 取得）。可先用「複製提問」貼到外部 LLM。"}
    start, errors = time.time(), []
    for model in dict.fromkeys(GEMINI_MODELS):
        left = LLM_TIME_BUDGET - (time.time() - start)
        if left < 8:
            break
        try:
            r = requests.post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": key},
                              json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}},
                              timeout=min(45, left))
            if r.ok:
                parts = ((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return {"answer": text, "model": model}
                errors.append(f"{model}: 空回應")
            else:
                errors.append(f"{model}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{model}: {type(e).__name__}")
    return {"error": "免費 LLM 目前忙碌或無法使用（" + "；".join(errors[:4]) + "）。請稍後再試，或用「複製提問」貼到外部 LLM。"}
