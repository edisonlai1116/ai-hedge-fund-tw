"""資產配置（股 / 債 / 現金）＋ 依市場狀態給調整步驟。

確定性規則，不是預測：
- 目標比例依恐懼貪婪 / VIX 調整。依據本站回測（2009 起）：恐懼時股票池未來報酬較高、貪婪時沒有明顯較差，
  所以「恐懼多放股票、貪婪不減股」——情緒只用來決定現金要不要多投入，不用來追高或殺低。
- 買賣標的完全沿用策略本身的訊號：賣出換股 / 減碼 → 低檔可買 → 持股加碼 → 動能新買（調整日）。
- 金額以台幣計；美元現金依匯率換算。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# 債券 ETF（美股）；台股代號 00 開頭且結尾 B 也視為債券 ETF
US_BOND_ETFS = {"TLT", "IEF", "IEI", "SHY", "BND", "AGG", "BNDX", "GOVT", "TLH", "VGIT", "VGLT", "VGSH", "LQD",
                "VCIT", "VCSH", "HYG", "JNK", "TIP", "SCHO", "SCHR", "SCHZ", "MUB", "EMB"}
# 貨幣市場 / 超短債：視同現金
CASH_LIKE_ETFS = {"SGOV", "BIL", "SHV", "USFR", "TFLO", "00865B.TW", "00719B.TW"}
OTHER_ETFS = {"GLD", "IAU", "SLV", "00635U.TW", "00708L.TW"}

TOLERANCE = 3.0      # 偏離目標 < 3 個百分點不動（避免來回交易成本）
MIN_TRADE_TWD = 10_000
# 生活用現金保留水位：總資產 2%、至少 NT$10 萬；只投入超過水位的現金（不會叫你把活存買光）。
# 回測（留 10% 現金年化約少 4 個百分點）線性推估，留 2% 約少 0.8 個百分點（推估，非獨立回測）。
CASH_RESERVE_PCT = 2.0
CASH_RESERVE_MIN_TWD = 100_000
MAX_NAME_PCT = 10.0   # 配置建議裡單檔新買上限（佔總資產 %）
MAX_NEW_MOMENTUM = 5  # 每市場最多列 5 檔動能新買（依排名），避免資金切太碎


def _shares(amount_twd: float, price: Optional[float], market: str, fx: float, up: bool = False) -> Optional[int]:
    """金額換算股數（賣出無條件進位、買進無條件捨去）。"""
    import math
    px = (price or 0) * (1.0 if market == "tw" else fx)
    if px <= 0:
        return None
    n = amount_twd / px
    return int(math.ceil(n) if up else math.floor(n))


def asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s in CASH_LIKE_ETFS:
        return "cash"
    if s in OTHER_ETFS:
        return "other"
    if s in US_BOND_ETFS:
        return "bond"
    if s.endswith((".TW", ".TWO")):
        code = s.split(".")[0]
        if code.startswith("00") and code.endswith("B"):
            return "bond"
    return "stock"


def target_mix(report: Dict) -> Dict:
    """目標股 / 債 / 現金（%）：使用者選擇「報酬最大化、不需緊急預備金」→ 全部放股票。
    依據 2026-10-10 回測（2017 起，低檔 30% / 動能 70%，現金年化 2%）：全額投入 美 52.3% / 台 49.2%；
    恐懼貪婪 > 75 留 15% 現金 49.3% / 45.9%；指數乖離 > 20% 留 15% —（美未觸發）/ 47.0%；VIX < 13 留 10% 51.2% / 49.3%；
    跌破 200 日線留 20% 51.6% / 48.1%（回撤少約 4.5 點）。沒有規則在兩市場都勝過全額投入 → 使用者選擇全額投入。"""
    sent = report.get("sentiment") or {}
    fg = ((sent.get("fear_greed") or {}).get("score"))
    vix = (sent.get("vix") or {}).get("value")
    mix = {"stock": 100, "bond": 0, "cash": 0}
    if (fg is not None and fg < 25) or (vix is not None and vix >= 30):
        label = "極度恐懼／恐慌"
        reasons = ["恐慌期：只買不賣（回測：恐慌時不賣、照常買進，報酬較高）。"]
    elif fg is not None and fg < 45:
        label = "恐懼"
        reasons = ["恐懼區歷史上偏有利買方：照規則全額投入。"]
    elif fg is not None and fg > 75:
        label = "極度貪婪"
        reasons = ["貪婪時不減股（回測：貪婪時賣出/等待反而較差），新買不追大長紅。"]
    else:
        label = "中性／貪婪"
        reasons = ["照規則全額投入。"]
    reasons.append("目標 100% 股票：回測 9 種依市場狀況留現金的規則（恐懼貪婪過熱、指數乖離、VIX 偏低、跌破 200 日線…），"
                   "沒有一種在美股與台股都比全額投入賺得多；停利賣掉創新高的股也較差。代價是回撤可能到 -35%~-40%。")
    if fg is not None:
        reasons.append(f"恐懼貪婪 {fg:.0f}" + (f"、VIX {vix:.1f}" if vix is not None else "") + "。")
    return {"mix": mix, "label": label, "reasons": reasons, "fear_greed": fg, "vix": vix}


def build_allocation(ev: Dict, report: Dict, cash_twd: float = 0.0, cash_usd: float = 0.0,
                     fx_usd_twd: float = 32.0) -> Dict:
    from src.strategy.ask import raise_cash_plan
    from src.strategy.momentum import ALLOCATION, LOWENTRY_SLOTS, TOP_N

    holdings = ev.get("holdings", [])
    cash_cash = float(cash_twd or 0) + float(cash_usd or 0) * fx_usd_twd
    amt = {"stock": 0.0, "bond": 0.0, "cash": cash_cash, "other": 0.0}
    by_mkt = {"us": 0.0, "tw": 0.0}
    for h in holdings:
        c = asset_class(h["symbol"])
        amt[c] += h["value_twd"] or 0
        if c == "stock":
            by_mkt[h["market"]] += h["value_twd"] or 0
    total = sum(amt.values())
    if total <= 0:
        return {"total_twd": 0, "current": {}, "target": {}, "steps": [], "notes": ["請先輸入持股或現金。"]}

    tm = target_mix(report)
    tgt = {k: total * v / 100 for k, v in tm["mix"].items()}
    reserve = max(total * CASH_RESERVE_PCT / 100, CASH_RESERVE_MIN_TWD)
    reserve = min(reserve, total)
    if reserve > tgt["cash"]:
        tgt["stock"] -= reserve - tgt["cash"]
        tgt["cash"] = reserve
    tgt_pct = {k: round(v / total * 100, 1) for k, v in tgt.items()}
    pct = {k: round(v / total * 100, 1) for k, v in amt.items()}
    panic = bool(ev.get("panic_no_sell"))
    rbi = ev.get("rebalance") or {}
    in_window = bool(ev.get("in_rebalance_window"))
    next_reb = ev.get("next_rebalance")

    def win(m: str) -> bool:
        return bool((rbi.get(m) or {}).get("in_window", in_window))

    def when_of(m: str) -> str:
        nx = (rbi.get(m) or {}).get("next", next_reb) or ""
        return "今天" if win(m) else f"下一步（約 {nx[5:].replace('-', '/')}）"
    steps: List[Dict] = []
    notes: List[str] = []

    # 1) 策略本身的賣出 / 減碼
    proceeds = 0.0
    swap_cash, swap_names = {"us": 0.0, "tw": 0.0}, {"us": [], "tw": []}   # 調整日才賣的錢：預先排好同一天要買什麼（換股預覽）
    plan = {x["symbol"]: x for x in raise_cash_plan(ev, report)}
    for h in holdings:
        if h["action"] == "賣出換股":
            if panic:
                notes.append(f"{h['symbol']} 系統建議賣出換股，但目前恐慌期「只買不賣」，先不賣。")
                continue
            when = when_of(h["market"])
            steps.append({"kind": "sell", "symbol": h["symbol"], "market": h["market"], "amount_twd": round(h["value_twd"]),
                          "shares": int(h["shares"]), "shares_note": f"全部 {h['shares']:,.0f} 股",
                          "when": when, "why": "跌出動能保留名單、目前最弱：賣出，錢轉入下方同一天的買進（換股）。"})
            if win(h["market"]):
                proceeds += h["value_twd"]
            else:
                swap_cash[h["market"]] += h["value_twd"]
                swap_names[h["market"]].append(h["symbol"].replace(".TWO", "").replace(".TW", ""))
        elif h["action"] == "減碼":
            trim = (plan.get(h["symbol"]) or {}).get("suggest_trim_twd") or 0
            if trim >= MIN_TRADE_TWD:
                n = h.get("trim_shares") or _shares(trim, h.get("price"), h["market"], fx_usd_twd, up=True)
                steps.append({"kind": "trim", "symbol": h["symbol"], "market": h["market"], "amount_twd": round(trim),
                              "shares": n, "shares_note": f"賣 {n:,} 股，留 {h['shares'] - n:,.0f} 股" if n else None,
                              "when": "今天", "why": h.get("reason") or "單檔過度集中：減碼到上限以下。"})
                proceeds += trim

    stock_after = amt["stock"] - proceeds
    cash_after = amt["cash"] + proceeds
    stock_gap = tgt["stock"] - stock_after          # > 0 要多買股票
    bond_gap = tgt["bond"] - amt["bond"]

    # 2) 股票過重：不為了配置比例賣股（與個股建議衝突、也違反「情緒不用來賣」的回測結論）；
    #    賣出只來自個股本身的訊號（賣出換股／集中度減碼），比例靠新資金與現金慢慢拉回。
    if stock_gap < -total * TOLERANCE / 100:
        notes.append(f"股票 {pct['stock']}% 高於目標 {tgt_pct['stock']}%：不為了比例賣股（會跟個股建議衝突，回測也顯示情緒偏高時賣股報酬較差）；"
                     f"之後的新資金與賣股所得先補現金／債券，直到比例回到目標附近。")

    # 3) 可投入資金：現金超過目標的部分（含賣股所得）
    budget = max(0.0, min(stock_gap, cash_after - tgt["cash"]))
    spent = 0.0
    if stock_gap > total * TOLERANCE / 100 and budget < MIN_TRADE_TWD:
        notes.append(f"股票比例低於目標 {tgt_pct['stock']}%，但現金沒有超過目標水位，可用新資金補。")
    if budget >= MIN_TRADE_TWD or sum(swap_cash.values()) >= MIN_TRADE_TWD:
        mk_total = sum(by_mkt.values())
        mshare = {m: (by_mkt[m] / mk_total if mk_total else 0.5) for m in by_mkt}
        tgt_mkt = {m: (stock_after + budget) * mshare[m] for m in by_mkt}
        low_cap = {m: tgt_mkt[m] * ALLOCATION["lowentry"] / LOWENTRY_SLOTS for m in by_mkt}
        mom_cap = {m: tgt_mkt[m] * ALLOCATION["momentum"] / TOP_N[m] for m in by_mkt}
        cands: List[Dict] = []
        for m in ("us", "tw"):
            for b in (ev.get("low_entry_buys") or {}).get(m, []):
                cands.append({"symbol": b["symbol"], "name": b.get("name"), "market": m, "cap": low_cap[m], "have": 0.0, "price": b.get("close"),
                              "when": "今天（分 2~3 批）", "spike": b.get("spike"),
                              "why": f"低檔可買：{b.get('low_label') or '長線贏家回撤'}（距高點 {b.get('dd_52w_pct')}%、Tier {b.get('quality_tier') or '—'}）。"})
        for h in holdings:
            if h["action"] in ("低檔加碼", "加碼") and asset_class(h["symbol"]) == "stock":
                cap = low_cap[h["market"]] if h["action"] == "低檔加碼" else mom_cap[h["market"]]
                dplan = (h.get("decision") or {}).get("plan") or {}
                if dplan.get("tranches"):          # 決策層的分批方案：今天只買第 1 筆，且要已在進場區
                    if not dplan.get("first_tranche_now"):
                        continue
                    cap = h["value_twd"] + dplan["tranches"][0]["amount_twd"]
                cands.append({"symbol": h["symbol"], "market": h["market"], "cap": cap, "have": h["value_twd"], "price": h.get("price"),
                              "when": "今天" if h["action"] == "低檔加碼" else when_of(h["market"]),
                              "spike": h.get("spike"), "why": f"系統建議「{h['action']}」：部位低於目標。"})
        for m in ("us", "tw"):
            for b in (ev.get("new_buys") or {}).get(m, [])[:MAX_NEW_MOMENTUM]:
                cands.append({"symbol": b["symbol"], "name": b.get("name"), "market": m, "cap": mom_cap[m], "have": 0.0, "price": b.get("close"),
                              "when": when_of(m), "spike": b.get("spike"),
                              "why": f"動能排名第 {b.get('rank')} 名、6 個月 {b.get('ret_6m_pct')}%：動能新買。"})
        left = budget
        for c in cands:
            if left < MIN_TRADE_TWD:
                break
            need = max(0.0, c["cap"] - c["have"])
            buy = min(need, left)
            momentum = "動能新買" in c["why"]
            if buy < MIN_TRADE_TWD and not momentum:  # 動能新買先保留，下面「空槽放動能」再補足金額
                continue
            why = c["why"]
            if c.get("spike"):
                why += " 近 3 日大長紅：不追，掛大漲前收盤價或等 3 日後。"
            steps.append({"kind": "buy", "symbol": c["symbol"], "name": c.get("name"), "market": c["market"],
                          "amount_twd": round(buy), "price": c.get("price"), "when": c["when"], "why": why})
            left -= buy
        # 空槽放動能：剩下的錢平均加到已列出的動能新買，單檔不超過總資產 MAX_NAME_PCT
        mom_steps = [st for st in steps if st["kind"] == "buy" and "動能新買" in st["why"]]
        while left >= MIN_TRADE_TWD and mom_steps:
            room = [st for st in mom_steps if st["amount_twd"] < total * MAX_NAME_PCT / 100 - MIN_TRADE_TWD]
            if not room:
                break
            each = left / len(room)
            for st in room:
                add = min(each, total * MAX_NAME_PCT / 100 - st["amount_twd"])
                st["amount_twd"] = round(st["amount_twd"] + add)
                left -= add
            for st in room:
                if "空槽放動能" not in st["why"]:
                    st["why"] += "（含低檔空槽資金：空槽放動能）"
        spent = max(0.0, budget - left)
        # 換股預覽：調整日賣出的錢 → 同一市場、同一天買進（持股加碼 → 動能新買；有餘再平均加碼）
        for mk_ in ("us", "tw"):
            left2 = swap_cash[mk_]
            if left2 < MIN_TRADE_TWD:
                continue
            tag = f"（用賣出 {'、'.join(swap_names[mk_])} 的錢）"
            by_sym = {st["symbol"]: st for st in steps if st["kind"] == "buy"}
            for c in [c for c in cands if c["market"] == mk_ and c["when"].startswith("下一步")]:
                if left2 < MIN_TRADE_TWD:
                    break
                st = by_sym.get(c["symbol"])
                have = c["have"] + (st["amount_twd"] if st else 0)
                buy = min(max(0.0, c["cap"] - have), left2)
                if buy < 1:
                    continue
                if st:
                    st["amount_twd"] = round(st["amount_twd"] + buy)
                else:
                    st = {"kind": "buy", "symbol": c["symbol"], "name": c.get("name"), "market": c["market"],
                          "amount_twd": round(buy), "price": c.get("price"), "when": c["when"], "why": c["why"]}
                    steps.append(st)
                    by_sym[c["symbol"]] = st
                if tag not in st["why"]:
                    st["why"] += tag
                left2 -= buy
            mom2 = [st for st in steps if st["kind"] == "buy" and st["market"] == mk_ and st["when"].startswith("下一步") and "動能新買" in st["why"]]
            while left2 >= MIN_TRADE_TWD and mom2:
                room = [st for st in mom2 if st["amount_twd"] < total * MAX_NAME_PCT / 100 - MIN_TRADE_TWD]
                if not room:
                    break
                each = left2 / len(room)
                for st in room:
                    add = min(each, total * MAX_NAME_PCT / 100 - st["amount_twd"])
                    st["amount_twd"] = round(st["amount_twd"] + add)
                    left2 -= add
                    if tag not in st["why"]:
                        st["why"] += tag
        steps[:] = [st for st in steps if st["kind"] != "buy" or st["amount_twd"] >= MIN_TRADE_TWD]
        for st in steps:
            if st["kind"] == "buy":
                n = _shares(st["amount_twd"], st.pop("price", None), st["market"], fx_usd_twd)
                st["shares"], st["shares_note"] = n, (f"約 {n:,} 股" if n else "金額不足 1 股")
        if left >= MIN_TRADE_TWD:
            notes.append(f"還有約 NT${left:,.0f} 未分配：依策略「空槽放動能」，下次調整日平均投入動能前段名單（策略精選分頁），或等新的低檔買點。")
        # 幣別：台股買單要台幣、美股買單要美元
        need_twd = sum(s["amount_twd"] for s in steps if s["kind"] == "buy" and s["market"] == "tw")
        need_usd_twd = sum(s["amount_twd"] for s in steps if s["kind"] == "buy" and s["market"] == "us")
        have_usd_twd = float(cash_usd or 0) * fx_usd_twd + sum(s["amount_twd"] for s in steps if s["kind"] in ("sell", "trim") and s["market"] == "us")
        if need_usd_twd > have_usd_twd + MIN_TRADE_TWD:
            steps.append({"kind": "fx", "symbol": "TWD→USD", "market": "us", "amount_twd": round(need_usd_twd - have_usd_twd),
                          "when": "買美股前", "why": f"美股買單需要約 US${need_usd_twd / fx_usd_twd:,.0f}，美元現金不足的部分換匯（匯率 {fx_usd_twd:.2f}）。"})
        have_twd = float(cash_twd or 0) + sum(s["amount_twd"] for s in steps if s["kind"] in ("sell", "trim") and s["market"] == "tw")
        if need_twd > have_twd + MIN_TRADE_TWD:
            steps.append({"kind": "fx", "symbol": "USD→TWD", "market": "tw", "amount_twd": round(need_twd - have_twd),
                          "when": "買台股前", "why": "台股買單的台幣不足，部分美元換回台幣（或減少台股買單）。"})

    # 4) 債券：只用「超過目標水位、且沒用在買股」的現金
    spare = max(0.0, cash_after - tgt["cash"] - spent)
    if bond_gap > total * TOLERANCE / 100:
        amt_b = min(bond_gap, spare)
        if amt_b >= MIN_TRADE_TWD:
            steps.append({"kind": "bond", "symbol": "債券 ETF", "market": "tw", "amount_twd": round(amt_b), "when": "分批",
                          "why": f"債券 {pct['bond']}% 低於目標 {tgt_pct['bond']}%：用多出的現金補投資等級債／公債 ETF（台幣如 00679B、00937B；美元如 BND、IEF），股災時當備用子彈。"})
        else:
            notes.append(f"債券 {pct['bond']}% 低於目標 {tgt_pct['bond']}%，但沒有多出的現金：之後的新資金可先補債券（約 NT${bond_gap:,.0f}）。")
    elif -bond_gap > total * 5 / 100:
        notes.append(f"債券 {pct['bond']}%：目標是全部股票，可在調整日把債券轉進動能／低檔買進名單。")
    if amt["cash"] <= tgt["cash"]:
        notes.insert(0, f"現金 NT${amt['cash']:,.0f} 在生活用保留水位 NT${tgt['cash']:,.0f}（總資產 {CASH_RESERVE_PCT:.0f}%、至少 NT${CASH_RESERVE_MIN_TWD:,.0f}）以內："
                        "今天不動用現金買股；買點有新資金再買，或等調整日用賣股的錢換股。")

    # 5) 每檔加碼建議附資金來源：只用多出的現金；不夠就跳過。
    #    回測（2017 起）：為了加碼去賣最弱的持股，報酬與回撤都比不賣差（美 48.9% vs 49.2%、台 46.0% vs 47.0%）。
    free = max(0.0, cash_after - tgt["cash"] - spent)
    planned = {st["symbol"]: st for st in steps if st["kind"] == "buy"}
    low_tgt = ev.get("low_entry_target_twd") or {}
    for h in holdings:
        t = h.get("today") or ""
        if not ((("加碼" in t) or t in ("左側分批第 1 筆", "等第 1 筆價位")) and not t.startswith(("月調日", "調整日", "下一步"))):
            continue
        dplan = (h.get("decision") or {}).get("plan") or {}
        want = (dplan["tranches"][0]["amount_twd"] if dplan.get("tranches") else
                max(0.0, (low_tgt.get(h["market"]) or h.get("target_twd") or 0) - (h["value_twd"] or 0)))
        h["add_twd"] = round(want) if want else None
        if h["symbol"] in planned:
            st = planned[h["symbol"]]
            h["funding_note"] = f"資金：用現金 {st['amount_twd']:,.0f}（{st.get('shares_note') or ''}）。"
        elif want < MIN_TRADE_TWD:
            # 部位已達低檔目標：不是加碼點，改成不用動（兩張卡片一致）
            h["today"], h["funding_note"] = "不用動", None
            h["today_reason"] = f"接近低檔價，但部位已達目標（約 NT${(low_tgt.get(h['market']) or 0):,.0f}）：不加碼。"
        elif free >= MIN_TRADE_TWD:
            use = min(want, free)
            free -= use
            n = _shares(use, h.get("price"), h["market"], fx_usd_twd)
            h["funding_note"] = f"資金：觸發時用現金約 NT${use:,.0f}" + (f"（約 {n:,} 股）" if n else "") + "。"
        else:
            exits_ = [x["symbol"] for x in holdings if (x.get("decision") or {}).get("group") == "EXIT"]
            h["funding_note"] = ("資金：現金不足 → 等新資金，或從退出候選 " + "、".join(exits_[:3]) + " 釋出"
                                 "（依基本面判斷的移轉，未經回測；純動能回測中「賣最弱來加碼」較差）。" if exits_ else
                                 "資金：現金不足、也沒有退出候選 → 這次跳過，不賣其他持股來湊；有新資金再加。")

    order = {"sell": 0, "trim": 1, "fx": 2, "buy": 3, "bond": 4}
    steps.sort(key=lambda s: (order.get(s["kind"], 9), -s["amount_twd"]))
    if not steps:
        notes.insert(0, "比例都在目標 ±3% 以內、系統也沒有買賣訊號：今天不用調整。")
    return {
        "total_twd": round(total),
        "current": {k: {"twd": round(v), "pct": pct[k]} for k, v in amt.items()},
        "target": {k: {"twd": round(tgt[k]), "pct": tgt_pct[k]} for k in tgt},
        "regime": {"label": tm["label"], "reasons": tm["reasons"], "fear_greed": tm["fear_greed"], "vix": tm["vix"]},
        "cash_input": {"twd": round(float(cash_twd or 0)), "usd": round(float(cash_usd or 0), 2)},
        "steps": steps,
        "notes": notes,
        "disclaimer": "規則化建議（依本站策略訊號與回測），不保證報酬、非投資建議；實際下單前請自行確認。",
    }
