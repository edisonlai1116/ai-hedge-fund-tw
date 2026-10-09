"""每日建議：每天都給一個明確結論（多數日子是「今天不用動」），規則與回測完全相同。

每日檢查（任何交易日都可能觸發）：
  1. 低檔布局新訊號：長線贏家剛跌破 52 週高點 -30% → 買進（point-in-time 驗證，持有 12 個月）。
  2. 接近低檔價：距觸發價 3% 以內 → 先列觀察、設好價位。
  3. 調整日（美股每 4 週＝每月前 3 個平日、台股每週一二）：動能前 N 名買進；跌出前 KEEP_N 名就賣，每次最多換 2 檔（2026-10-10 回測）。
  4. 市場情緒：恐懼貪婪 < 25 → 調整日只買不賣（回測驗證）；其餘情緒狀態只提示，不改變買賣規則——
     回測顯示「貪婪時賣出 / 等恐懼才投入 / 恐慌時放寬低檔門檻」都會降低報酬。
  5. 類股情緒：過熱 / 超賣的族群（資訊，用來理解低檔訊號是個股問題還是整個族群一起跌）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.strategy.momentum import (KEEP_N, LIMIT_VALID_DAYS, LOWENTRY_DD, LOWENTRY_HOLD_MONTHS, PANIC_NO_SELL_FG,
                                   SPIKE_DAYS, SPIKE_PCT, TOP_N, TPE, is_rebalance_window, next_rebalance,
                                   rebalance_info, MAX_SWAPS, REBALANCE_EVERY, REBALANCE_LABEL)
from src.strategy.momentum import low_view
from src.strategy.sentiment import EXTREME_FEAR, EXTREME_GREED, VIX_PANIC

NEW_SIGNAL_DAYS = 3        # 進入低檔區 ≤ 3 個交易日 → 視為「新訊號」（容許排程漏跑）
NEAR_TRIGGER_PCT = 3.0     # 距低檔觸發價 3% 以內 → 觀察
MARKET_LABEL = {"us": "美股", "tw": "台股"}


def _study_row(backtest: Optional[Dict], market: str, fg: Optional[float]) -> Optional[Dict]:
    try:
        rows = backtest["markets"][market]["sentiment_study"]["rows"]
    except Exception:
        return None
    if fg is None:
        return None
    return next((r for r in rows if r["lo"] <= fg < r["hi"]), None)


def _sentiment_guidance(sent: Dict, backtest: Optional[Dict]) -> List[str]:
    fg = (sent.get("fear_greed") or {}).get("score")
    vix = sent.get("vix") or {}
    out = []
    st = _study_row(backtest, "us", fg)
    hist = (f"歷史上（2009 起）恐懼貪婪在「{st['bucket']}」時，AI 股池未來 3 個月平均 {st.get('pool_63', 0):+.1f}%、"
            f"大盤 {st.get('bench_63', 0):+.1f}%。") if st else ""
    if fg is None:
        out.append("恐懼貪婪資料暫缺，照一般規則執行。")
    elif fg < PANIC_NO_SELL_FG:
        out.append(f"極度恐懼（{fg:.0f}）：這次調整「只買不賣」，不要在恐慌時砍股票；低檔訊號照常買進。" + hist)
    elif fg < 45:
        out.append(f"恐懼（{fg:.0f}）：歷史上偏有利於買方；照規則執行，不需要額外動作。" + hist)
    elif fg > EXTREME_GREED:
        out.append(f"極度貪婪（{fg:.0f}）：不加槓桿、不追加規則以外的部位；但回測顯示貪婪不代表會跌，不建議因此賣出。" + hist)
    else:
        out.append(f"情緒{'偏貪婪' if fg > 55 else '中性'}（{fg:.0f}）：照規則執行。" + hist)
    if vix.get("value") and vix["value"] >= VIX_PANIC:
        out.append(f"VIX {vix['value']:.1f} ≥ {VIX_PANIC:.0f}（恐慌）：歷史上 VIX 30~40 之後 3 個月 AI 股池平均約 +13%；不要恐慌賣出。")
    if vix.get("backwardation"):
        out.append(f"VIX 期限結構逆價差（VIX {vix.get('value')} > 3 個月 {vix.get('vix3m')}）：短線恐慌升溫，波動會放大。")
    return out


DD_LABEL = {"SPECULATIVE_DE_RATING": "投機股預期修正", "VALUATION_RESET": "估值修正（仍不便宜）", "UNKNOWN": "回撤原因不明",
            "FUNDAMENTAL_DAMAGE": "基本面受損"}


def _market_actions(market: str, mk: Dict, sent: Dict, rebalance: bool, panic: bool, dd_types: Optional[Dict] = None) -> Dict:
    rows = mk.get("rows", [])
    by = {r["symbol"]: r for r in rows}
    actions: List[Dict] = []
    watch: List[Dict] = []

    def base(r: Dict) -> Dict:
        return {k: r.get(k) for k in ("symbol", "name", "close", "rank", "dd_52w_pct", "ret_3y_pct", "high_52w",
                                      "low_entry_price", "spike", "bounce_20d_pct", "virattt")}

    def bounce_txt(r: Dict) -> str:
        b = r.get("bounce_20d_pct")
        return "" if b is None else ("，尚未反彈（距 20 日低點 {:+.0f}%）".format(b) if b < 5 else "，已從 20 日低點反彈 {:+.0f}%".format(b))

    dd_types = dd_types or {}

    def dd_of(sym: str) -> Dict:
        return dd_types.get(sym) or {"type": "UNKNOWN", "why": "尚未做回撤分類"}

    low_rows = [by[s] for s in mk.get("low_entry", []) if s in by]
    low_rows.sort(key=lambda r: (r.get("bounce_20d_pct") is None, r.get("bounce_20d_pct") or 0))   # 還沒漲的排前面
    for r in low_rows:
        days = r.get("low_entry_days")
        sp = r.get("spike")
        lv = r.get("low_view") or low_view(r, {"drawdown_types": dd_types, "markets": {market: mk}}, mk.get("universe_size")) or {}
        rec = lv.get("recommendation")
        if rec not in ("BUY", "BUY_STAGED"):
            watch.append({**base(r), "type": "low_entry_not_buyable", "drawdown_type": lv.get("drawdown_type"),
                          "low_state": lv.get("state"), "quality_tier": lv.get("tier"),
                          "action": lv.get("label") or "低檔觀察（不直接買）",
                          "reason": f"價格跌深（距高點 {r.get('dd_52w_pct'):.0f}%）、排名第 {r.get('rank')} 名（Tier {lv.get('tier')}）："
                                    + (lv.get("why") or "尚未做回撤分類，不自動買。")})
            continue
        if sp:
            lim = sp.get("limit_price")
            actions.append({**base(r), "type": "low_entry_limit", "action": f"不追，掛 {lim} 等拉回",
                            "reason": f"在低檔區（距高點 {r.get('dd_52w_pct'):.0f}%），但 {sp['date']} 單日大漲 +{sp['gain_pct']}%："
                                      f"不買剛噴完的大長紅，掛大漲前收盤價 {lim}，{LIMIT_VALID_DAYS} 個交易日內沒回到就放棄。"})
            continue
        dd = dd_of(r["symbol"])
        if days is not None and days <= NEW_SIGNAL_DAYS and not r.get("low_entry_reentry"):
            actions.append({**base(r), "type": "low_entry_new", "action": f"買進（{lv.get('label')}）",
                            "drawdown_type": dd["type"], "low_state": lv.get("state"), "quality_tier": lv.get("tier"),
                            "reason": f"{lv.get('why')} 長線贏家（3 年 {r.get('ret_3y_pct'):+.0f}%）剛跌破 52 週高點 {r.get('high_52w')} 的 -30%"
                                      f"（目前 {r.get('dd_52w_pct'):.0f}%，第 {days} 天）：新低檔訊號，買進後持有 {LOWENTRY_HOLD_MONTHS} 個月。"})
        else:
            again = "（短暫反彈出去後又跌回，不算新訊號）" if r.get("low_entry_reentry") and days is not None and days <= NEW_SIGNAL_DAYS else ""
            watch.append({**base(r), "type": "low_entry_holding", "action": f"{lv.get('label')}（已在低檔區）", "drawdown_type": dd["type"],
                          "low_state": lv.get("state"), "quality_tier": lv.get("tier"),
                          "reason": f"在低檔區 {days if days is not None else '多'} 天{again}（距高點 {r.get('dd_52w_pct'):.0f}%{bounce_txt(r)}）；"
                                    + (lv.get("why") or "") + "還沒買、且低檔槽位未滿可分批（優先買還沒反彈的）；已買者續抱。"})
    for s in mk.get("low_entry_watch", []):
        r = by.get(s)
        if not r or r.get("low_entry_price") is None or not r.get("close"):
            continue
        gap = (r["close"] / r["low_entry_price"] - 1) * 100
        if gap <= NEAR_TRIGGER_PCT:
            watch.append({**base(r), "type": "near_trigger", "action": f"接近低檔價（差 {gap:.1f}%）",
                          "reason": f"跌到 {r['low_entry_price']}（52 週高點 -30%）就是低檔買點；可先設好價位提醒。"})
    if rebalance:
        for r in rows[:TOP_N[market]]:
            sp = r.get("spike")
            if sp:
                actions.append({**base(r), "type": "rebalance_wait", "action": "調整日：等大長紅過後再買",
                                "reason": f"排名第 {r['rank']} 名，但 {sp['date']} 單日大漲 +{sp['gain_pct']}%：沒持有的先不追，"
                                          f"等大漲超過 {SPIKE_DAYS} 個交易日、且仍在前 {TOP_N[market]} 名再買；已持有續抱。"})
            else:
                actions.append({**base(r), "type": "rebalance_buy", "action": "調整日：動能買進／續抱",
                                "reason": f"排名第 {r['rank']} 名（前 {TOP_N[market]} 名）；沒持有就等權買進。"})
        if panic:
            actions.append({"type": "panic_hold", "symbol": "—", "action": "這次調整只買不賣",
                            "reason": f"恐懼貪婪 < {PANIC_NO_SELL_FG:.0f}：跌出前 {KEEP_N[market]} 名的持股暫不賣，等情緒回穩的下次調整再換。"})
    for r in rows[:KEEP_N[market]]:
        ign = r.get("ignition")
        if ign and (ign.get("ignition_days_ago") or 9) <= 1:
            watch.append({**base(r), "type": "ignition", "action": "🔥 爆量長紅點火",
                          "reason": f"動能第 {r['rank']} 名、單日 +{ign.get('ignition_gain_pct')}%、量 {ign.get('ignition_volume_ratio')} 倍"
                                    "（事件提醒；主規則仍是調整日換股）。"})

    prefix = "美股·" if market == "us" else "台股·"
    groups = [g for g in (sent.get("sectors") or []) if g.get("kind") == "group" and str(g.get("key", "")).startswith(prefix)]
    hot = [g["name"].replace(prefix, "") for g in groups if g["mood"] == "過熱"]
    cold = [g["name"].replace(prefix, "") for g in groups if g["mood"] == "超賣"]

    n_act = sum(1 for a in actions if a["type"] not in ("panic_hold", "low_entry_limit", "rebalance_wait"))
    n_lim = sum(1 for a in actions if a["type"] == "low_entry_limit")
    if n_act:
        level = "action"
        parts = []
        nn = sum(1 for a in actions if a["type"] == "low_entry_new")
        if nn:
            parts.append(f"{nn} 檔新低檔買點")
        if rebalance:
            parts.append("調整日")
        if n_lim:
            parts.append(f"{n_lim} 檔掛限價等拉回")
        summary = "今天要操作：" + "、".join(parts)
    elif n_lim:
        level = "watch"
        summary = f"今天不追高：{n_lim} 檔大漲後掛限價等拉回" + (f"；另 {len(watch)} 檔觀察" if watch else "")
    elif watch:
        level = "watch"
        summary = f"今天不用買賣；{len(watch)} 檔列入觀察"
    else:
        level = "hold"
        summary = "今天不用動：沒有新訊號、非調整日"
    vr = [r for r in rows if r.get("virattt")]
    vr.sort(key=lambda r: r["virattt"]["score"], reverse=True)
    pick = lambda xs: [{k: r.get(k) for k in ("symbol", "name", "close", "rank", "day_change_pct", "virattt")} for r in xs]
    virattt = {"bullish": pick(vr[:10]), "bearish": pick(vr[::-1][:10]),
               "counts": {sig: sum(1 for r in vr if r["virattt"]["signal"] == sig) for sig in ("bullish", "neutral", "bearish")},
               "used_in_ranking": market == "tw"}
    return {"level": level, "summary": summary, "actions": actions, "watch": watch[:15], "virattt": virattt,
            "hot_groups": hot, "oversold_groups": cold, "momentum_share": mk.get("momentum_share")}


def build_daily_advice(report: Dict, backtest: Optional[Dict] = None) -> Dict:
    now = datetime.now(TPE)
    sent = report.get("sentiment") or {}
    fg = (sent.get("fear_greed") or {}).get("score")
    panic = fg is not None and fg < PANIC_NO_SELL_FG
    rebalance = is_rebalance_window(now.date())
    markets = {m: _market_actions(m, mk, sent, is_rebalance_window(now.date(), m), panic, report.get("drawdown_types"))
               for m, mk in (report.get("markets") or {}).items()}
    levels = [d["level"] for d in markets.values()]
    if "action" in levels:
        head = "今天有操作：" + "；".join(f"{MARKET_LABEL[m]} {d['summary'].replace('今天要操作：', '')}"
                                        for m, d in markets.items() if d["level"] == "action")
    elif "watch" in levels:
        head = "今天不用買賣，留意觀察名單"
    else:
        head = "今天不用動"
    fg_txt = f"恐懼貪婪 {fg:.0f}（{sent['fear_greed'].get('label')}）" if fg is not None else ""
    vix = (sent.get("vix") or {}).get("value")
    head += "｜" + "、".join(x for x in (fg_txt, f"VIX {vix:.1f}" if vix else "") if x)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "headline": head,
        "panic_no_sell": panic,
        "in_rebalance_window": rebalance,
        "next_rebalance": next_rebalance(now.date()),
        "rebalance": rebalance_info(now.date()),
        "sentiment_guidance": _sentiment_guidance(sent, backtest),
        "markets": markets,
        "rules": [
            {"rule": f"低檔：跌破 52 週高點 {LOWENTRY_DD:.0%} 只是「價格跌深」。排名 Tier A + 回撤屬折價/暫時衝擊 → 可買；"
                      f"Tier B → 分批（半個部位）；Tier C → 低檔觀察；Tier D → 不因跌深而買（需獨立證據才是反轉候選）；"
                      f"基本面受損 → 不買。持有 {LOWENTRY_HOLD_MONTHS} 個月", "check": "每天"},
            {"rule": f"動能輪動：買前 N 名、跌出 KEEP_N 名就換（每次最多 {MAX_SWAPS['us']} 檔，最弱先換）",
             "check": f"美股{REBALANCE_LABEL[REBALANCE_EVERY['us']]}（每月前 3 個平日）、台股{REBALANCE_LABEL[REBALANCE_EVERY['tw']]}（週一、二）"},
            {"rule": "低檔區沒用到的資金放動能名單（不留現金）", "check": "每天（低檔訊號出現時從動能部位挪錢）"},
            {"rule": f"恐懼貪婪 < {PANIC_NO_SELL_FG:.0f}：調整日只買不賣", "check": "調整日"},
            {"rule": f"不追大長紅：近 {SPIKE_DAYS} 日單日漲 ≥ {SPIKE_PCT:.0%} → 低檔股掛大漲前收盤價（{LIMIT_VALID_DAYS} 日有效）、動能股等 {SPIKE_DAYS} 日後再買",
             "check": "每天"},
        ],
        "disclaimer": "規則化訊號，回測有倖存者偏差，非投資建議。",
    }


def append_log(advice: Dict, path: str, keep: int = 400) -> None:
    """每日建議歷史（同一天只保留最後一筆）。"""
    line = {
        "date": advice["date"], "headline": advice["headline"], "panic_no_sell": advice["panic_no_sell"],
        "markets": {m: {"level": d["level"], "summary": d["summary"],
                        "buys": [a["symbol"] for a in d["actions"] if a["type"] == "low_entry_new"]}
                    for m, d in advice["markets"].items()},
    }
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
    rows = [r for r in rows if r.get("date") != line["date"]] + [line]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows[-keep:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
