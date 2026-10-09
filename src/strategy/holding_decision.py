"""持股決策層：四分組（核心持有／左側布局候選／等待確認／退出候選）、獨立加碼評分與分批方案、
基本面複查賣出、回收現金方案、資料可信度。

設計原則
- 不讓單一指標決定買賣：排名跌出保留名單只「觸發複查」；帳面虧損、跌幅都不單獨構成買或賣。
- 只用系統真的有的資料：機會評分（docs/data/opportunity.json，或即時 src.ranking.engine.rank_stocks）的
  品質／成長／財報加速／估值分數、回撤分類、確認訊號、衝擊證據、ATR 進場區、波動度；策略排名與 3 年報酬。
  沒有的（估值歷史百分位、同業估值、分析師目標價、校準過的預期報酬）一律標「資料不足」，不補造。
- 回測基準：momentum.evaluate_holdings(decisions=False) 是原規則；本層改動的列都保留 baseline_action。
  基本面沒有歷史時點資料 → 本層的複查／加碼規則無法回測，輸出時明確標示。

規則優先級（高 → 低）
1. 核心 ETF／資料不足 → 不套用個股規則（資料不足時沿用回測規則，並標示）。
2. 集中度：單檔 > 總資產上限 → 部分減碼（不是清倉）；長期邏輯仍在 → 仍屬核心持有。
3. 長期邏輯破壞（回撤分類＝基本面受損、或衝擊損害 ≥ 75、或品質與成長皆 < 35）且動能也弱 → 退出候選（全數退出）。
4. 排名跌出保留名單 → 複查：基本面強且下跌屬暫時衝擊／折價 → 續抱、停止加碼；否則照回測規則換股。
5. 價格回落（距高點 ≤ -15% 或帳面虧損 ≤ -10%）→ 獨立加碼評分 → 左側布局可買／低檔分批／等待確認／不宜加碼。
6. 其餘 → 核心持有（續抱、不加碼）或等待確認（基本面偏弱）。
"""
from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from typing import Dict, Iterable, List, Optional

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "data")
_CACHE: Dict[str, tuple] = {}          # symbol -> (ts, row or None)
_TTL = 24 * 3600

GROUP_LABEL = {"CORE": "核心持有", "LEFT_SIDE": "左側布局候選", "WAIT": "等待確認", "EXIT": "退出候選"}
DECISION_LABEL = {"EXIT_ALL": "全數退出", "TRIM": "部分減碼", "HOLD_NO_ADD": "續抱但停止加碼",
                  "HOLD_CONSIDER_ADD": "續抱並考慮左側分批", "HOLD": "續抱"}
GRADES = ("左側布局可買", "低檔分批", "等待確認", "不宜加碼", "資料不足，不能判定")
CAUSE_SCORE = {"TEMPORARY_SHOCK": 100, "FUNDAMENTAL_DISCOUNT": 90, "VALUATION_RESET": 55, "NONE": 50,
               "UNKNOWN": 30, "SPECULATIVE_DE_RATING": 15, "FUNDAMENTAL_DAMAGE": 0}
CAUSE_LABEL = {"TEMPORARY_SHOCK": "暫時衝擊（大盤／產業／技術面帶動）", "FUNDAMENTAL_DISCOUNT": "基本面未破壞的折價",
               "VALUATION_RESET": "昂貴估值的修正（尚未便宜）", "NONE": "回落不到 20%，不做回撤分類",
               "UNKNOWN": "證據不足，原因未判定", "SPECULATIVE_DE_RATING": "投機股預期修正（虧損或營收極小）",
               "FUNDAMENTAL_DAMAGE": "基本面受損（財報／預估下修／公司特有利空）"}
CONF_LABEL = {"EARNINGS_ACCELERATION": "財報加速", "FUNDAMENTAL_RECOVERY": "EPS 預估上修", "CATALYST": "催化事件",
              "POSITIVE_REVERSAL": "止跌反轉（20 日 +5% 且站上 50 日線）"}
TIER_SCORE = {"A": 100, "B": 70, "C": 40, "D": 10}
ALWAYS_MISSING = ["估值相對自身歷史的百分位", "同業估值比較", "分析師目標價", "校準過的 12 個月預期報酬"]
NOT_BACKTESTED = "基本面只有即時資料、沒有歷史時點紀錄 → 這項判斷無法回測。"


# --------------------------------------------------------------------------- 資料
def _load_opportunity() -> Dict[str, Dict]:
    try:
        with open(os.path.join(_DOCS, "opportunity.json"), encoding="utf-8") as f:
            rows = json.load(f).get("ranking") or []
        return {r["ticker"]: r for r in rows if r.get("ticker") and r.get("scores")}
    except Exception:
        return {}


_LOCK = threading.Lock()
PENDING: set = set()                   # 背景抓取中的代號（頁面顯示「載入中」）


def _fetch_live(need: List[str]) -> None:
    """背景執行：一次只跑一個（免費主機記憶體只有 512MB），不抓新聞以省資源。"""
    if not _LOCK.acquire(blocking=False):
        return
    try:
        from src.ranking.engine import rank_stocks
        for i in range(0, len(need), 10):
            chunk = need[i:i + 10]
            try:
                res = rank_stocks(chunk, live_news=False)
                got = {r.get("ticker"): r for r in res.get("ranking", []) if r.get("scores")}
            except Exception:
                got = {}
            now = time.time()
            for s in chunk:
                r = got.get(s)
                if r:
                    r.pop("fundamentals", None)
                _CACHE[s] = (now, r)
                PENDING.discard(s)
            del res
            gc.collect()
    finally:
        for s in need:
            PENDING.discard(s)
        _LOCK.release()


def load_fundamentals(symbols: Iterable[str], live: bool = True, wait: bool = False) -> Dict[str, Dict]:
    """先用每日機會評分；沒有的持股在背景即時算（rank_stocks，快取 1 天），這次先回傳已有的。
    wait=True 時同步等待（測試／離線批次用）。抓不到就不放（＝資料不足）。"""
    syms = list(dict.fromkeys(symbols))
    base = _load_opportunity()
    out = {s: base[s] for s in syms if s in base}
    now = time.time()
    need = []
    for s in syms:
        if s in out:
            continue
        c = _CACHE.get(s)
        if c and now - c[0] < _TTL:
            if c[1]:
                out[s] = c[1]
        elif s not in PENDING:
            need.append(s)
    if need and live and os.environ.get("HOLDING_DECISION_LIVE", "1") != "0":
        if wait:
            _fetch_live(need)
            out.update({s: _CACHE[s][1] for s in need if s in _CACHE and _CACHE[s][1]})
        elif not _LOCK.locked():
            PENDING.update(need)
            threading.Thread(target=_fetch_live, args=(need,), daemon=True).start()
    return out


def _n(v) -> str:
    """數字顯示（None → —），避免格式化 None 造成整個評估失敗。"""
    return "—" if v is None else f"{v:.0f}"


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _facts(h: Dict, r: Optional[Dict], f: Optional[Dict], report: Dict) -> Dict:
    sc = (f or {}).get("scores") or {}
    dt = report.get("drawdown_types", {}).get(h["symbol"]) or {}
    dd_type = (f or {}).get("drawdown_type") or dt.get("type")
    dd_why = ((f or {}).get("drawdown") or {}).get("why") or dt.get("why")
    shock = (f or {}).get("shock") or {}
    fu = ((f or {}).get("risk_detail") or {}).get("fundamental_uncertainty") or {}
    return {
        "has_f": bool(sc) and (_f(sc.get("quality")) is not None or _f(sc.get("growth")) is not None),
        "q": _f(sc.get("quality")), "g": _f(sc.get("growth")), "acc": _f(sc.get("earnings_acceleration")),
        "val": _f(sc.get("valuation")), "cat": _f(sc.get("catalyst")),
        "dd_type": dd_type, "dd_why": dd_why,
        "conf": list((f or {}).get("confirmations") or dt.get("confirmations") or []),
        "damage": _f(shock.get("fundamental_damage_score")) or 0.0,
        "shock_evidence": shock.get("evidence") or [],
        "speculative": bool(fu.get("speculative")),
        "fu_evidence": fu.get("evidence") or [],
        "vol": _f((f or {}).get("vol_63")) or (_f((r or {}).get("vol_ann_pct")) or 0) / 100 or None,
        "downside": _f((f or {}).get("downside_risk")),
        "entry": (f or {}).get("entry") or {},
        "val_class": (f or {}).get("valuation_class"),
        "rec_mean": _f((((f or {}).get("positioning") or {}).get("metrics") or {}).get("recommendation_mean")),
        "rank": (r or {}).get("rank") or h.get("rank"),
        "tier": (r or {}).get("quality_tier"),
        "ret_3y": _f((r or {}).get("ret_3y_pct")),
        "dd52": _f((r or {}).get("dd_52w_pct") if r else h.get("dd_52w_pct")),
        "lt_broken": bool((r or {}).get("long_term_broken")),
        "low_entry": bool((r or {}).get("low_entry")),
        "pnl": _f(h.get("pnl_pct")),
    }


# --------------------------------------------------------------------------- 判斷
def thesis_status(x: Dict, keep_n: int) -> tuple:
    """長期投資邏輯：intact / weak / broken / unknown，附理由。"""
    if not x["has_f"]:
        if x["lt_broken"]:
            return "broken", "3 年報酬 ≤ 0（長線趨勢破壞）；基本面資料不足"
        return "unknown", "基本面資料不足，不能判定"
    q, g = x["q"], x["g"]
    if x["dd_type"] == "FUNDAMENTAL_DAMAGE" or x["damage"] >= 75:
        return "broken", "回撤分類為基本面受損" if x["dd_type"] == "FUNDAMENTAL_DAMAGE" else f"公司特有衝擊損害分數 {x['damage']:.0f}"
    if q is not None and g is not None and q < 35 and g < 35:
        return "broken", f"品質 {_n(q)}、成長 {_n(g)} 都偏低"
    if x["lt_broken"] and (x["rank"] or 0) > keep_n:
        return "broken", "動能跌出保留名單且 3 年報酬 ≤ 0"
    weak = []
    if x["speculative"] or x["dd_type"] == "SPECULATIVE_DE_RATING":
        weak.append("投機性高（虧損或營收極小）")
    if q is not None and q < 50:
        weak.append(f"品質 {_n(q)} 偏低")
    if g is not None and g < 40:
        weak.append(f"成長 {_n(g)} 偏低")
    if x["damage"] >= 50:
        weak.append(f"衝擊損害 {x['damage']:.0f}")
    if weak:
        return "weak", "、".join(weak)
    return "intact", f"品質 {_n(q)}、成長 {_n(g)}" + (f"、回撤屬{CAUSE_LABEL.get(x['dd_type'], x['dd_type'])}" if x["dd_type"] not in (None, "NONE") else "")


def strong_core(x: Dict, status: str) -> bool:
    return (status == "intact" and (x["q"] or 0) >= 60 and (x["g"] or 0) >= 50
            and x["dd_type"] in ("NONE", "TEMPORARY_SHOCK", "FUNDAMENTAL_DISCOUNT"))


def add_score(x: Dict, weight_pct: float, cap_pct: float) -> Dict:
    """獨立的加碼評分（0~100），不等於動能排名。缺的元件不計、權重重新分配，並列出缺項。"""
    comp = {
        "fundamentals": ((x["q"] + x["g"]) / 2 if x["q"] is not None and x["g"] is not None else None, 0.25, "基本面（品質＋成長）"),
        "valuation": (x["val"], 0.20, "估值吸引力"),
        "acceleration": (x["acc"], 0.10, "財報加速／營運動能"),
        "cause": (CAUSE_SCORE.get(x["dd_type"]) if x["dd_type"] else None, 0.15, "下跌原因"),
        "confirmations": (min(100.0, 25.0 * len(x["conf"])) if x["has_f"] else None, 0.10, "反轉／改善證據"),
        "risk_reward": (100 * (1 - min(1.0, x["downside"] / 0.5)) if x["downside"] is not None else None, 0.10, "下行風險"),
        "position_room": (100 * max(0.0, 1 - weight_pct / cap_pct), 0.05, "部位空間"),
        "opportunity_cost": (TIER_SCORE.get(x["tier"]) if x["tier"] else None, 0.05, "資金機會成本（排名相對位置）"),
    }
    have = {k: v for k, v in comp.items() if v[0] is not None}
    wsum = sum(v[1] for v in have.values())
    score = sum(v[0] * v[1] for v in have.values()) / wsum if wsum else None
    return {"score": round(score, 1) if score is not None else None,
            "components": {v[2]: round(v[0], 1) for v in have.values()},
            "missing": [v[2] for k, v in comp.items() if v[0] is None]}


def add_grade(x: Dict, a: Dict, status: str, weight_pct: float, cap_pct: float) -> tuple:
    if not x["has_f"] or a["score"] is None:
        return "資料不足，不能判定", "沒有基本面／估值資料，不能判定能不能攤平。"
    if status == "broken":
        return "不宜加碼", "長期投資邏輯已破壞，跌深不是便宜。"
    if weight_pct >= cap_pct:
        return "不宜加碼", f"部位已占總資產 {weight_pct:.1f}%，達單檔上限 {cap_pct:.0f}%。"
    if x["dd52"] is not None and x["dd52"] > -15:
        return "不宜加碼", (f"帳面虧損來自買進成本較高，股價距 52 週高點只回落 {abs(x['dd52']):.0f}%：價格不在低檔，"
                         "帳面虧損不代表便宜。")
    s = a["score"]
    if x["speculative"] or x["dd_type"] in ("SPECULATIVE_DE_RATING", "UNKNOWN"):
        return ("等待確認", "下跌原因未判定或投機性高：先等證據。") if s >= 40 else ("不宜加碼", "證據不足且加碼評分偏低。")
    if s >= 70 and x["dd_type"] in ("TEMPORARY_SHOCK", "FUNDAMENTAL_DISCOUNT") and x["conf"] and status == "intact":
        return "左側布局可買", "基本面完整、下跌屬暫時衝擊或折價，且已有改善證據。"
    if s >= 55 and status in ("intact", "weak") and x["dd_type"] != "VALUATION_RESET":
        return "低檔分批", "長期邏輯尚可，但估值或短期基本面仍有不確定性：只適合小量分批。"
    if s >= 55 and x["dd_type"] == "VALUATION_RESET":
        return "等待確認", "是昂貴估值的修正、還沒便宜：等估值回到合理或出現改善證據。"
    if s >= 40:
        return "等待確認", "有潛力，但證據不足，不急著攤平。"
    return "不宜加碼", "加碼評分偏低（估值、基本面或風險報酬不佳）。"


def tranche_plan(x: Dict, grade: str, a: Dict, value_twd: float, total_twd: float, cap_pct: float,
                 price: Optional[float], fx: float, market: str) -> Optional[Dict]:
    """分批方案：比例依加碼評分（確定性）與波動計算；價位用機會評分的 ATR 進場區（不是補造的目標價）。"""
    if grade not in ("左側布局可買", "低檔分批") or not total_twd:
        return None
    room = max(0.0, total_twd * cap_pct / 100 - value_twd)
    budget = room * (1.0 if grade == "左側布局可買" else 0.5)
    if budget < 10_000:
        return {"budget_twd": round(budget), "note": "距單檔上限的空間不到 NT$1 萬：不需再加。", "tranches": []}
    cert = max(0.0, min(1.0, ((a["score"] or 55) - 55) / 45))
    volf = max(0.0, min(1.0, (x["vol"] or 0.4) / 0.8))
    first = max(0.15, min(0.45, 0.25 + 0.20 * cert - 0.10 * volf))
    second = (1 - first) * 0.5
    third = 1 - first - second
    e = x["entry"] or {}
    z1, z2, z3 = e.get("zone_1") or {}, e.get("zone_2") or {}, e.get("zone_3") or {}
    atr = _f(e.get("atr"))
    ok_now = bool((e.get("condition") or {}).get("ok"))
    cur = "（美元）" if market == "us" else ""

    def amt(p):
        return round(budget * p)

    def sh(p):
        px = (price or 0) * (1.0 if market == "tw" else fx)
        return int(budget * p / px) if px > 0 else None

    t1 = {"n": 1, "pct": round(first * 100), "amount_twd": amt(first), "shares": sh(first),
          "trigger": "現價已在可進場區，可以開始" if ok_now else
          (f"回到第 1 進場區 {z1.get('low')}~{z1.get('high')}{cur}（{z1.get('basis', '')}）" if z1 else "等價格回到進場區")}
    t2 = {"n": 2, "pct": round(second * 100), "amount_twd": amt(second), "shares": sh(second),
          "trigger": (f"跌到第 2 進場區 {z2.get('low')}~{z2.get('high')}{cur}（{z2.get('basis', '')}）" if z2 else "再跌約 1 個 ATR")
          + "，或出現新的確認訊號（財報加速／止跌反轉）"}
    t3 = {"n": 3, "pct": round(third * 100), "amount_twd": amt(third), "shares": sh(third),
          "trigger": (f"跌到第 3 進場區 {z3.get('low')}~{z3.get('high')}{cur}（{z3.get('basis', '')}）" if z3 else "再跌約 2 個 ATR")
          + "，且下一次財報沒有讓回撤分類轉成基本面受損"}
    stop = None
    if z3.get("low") is not None and atr:
        stop = round(float(z3["low"]) - atr, 2)
    invalid = ["回撤分類轉為「基本面受損」或公司特有衝擊損害 ≥ 75"]
    if stop:
        invalid.append(f"收盤跌破 {stop}{cur}（第 3 進場區下緣再下 1 個 ATR）")
    invalid.append("3 年報酬轉為 ≤ 0 且跌出保留名單")
    return {"budget_twd": round(budget), "tranches": [t1, t2, t3], "invalidation": invalid,
            "basis": f"總預算＝距單檔上限 {cap_pct:.0f}% 的空間" + ("" if grade == "左側布局可買" else " × 50%（不確定性較高）")
            + f"；第 1 筆比例依加碼評分 {_n(a['score'])}、年化波動 {(x['vol'] or 0) * 100:.0f}% 計算。",
            "first_tranche_now": ok_now}


def wait_detail(x: Dict, a: Dict) -> Dict:
    missing = []
    if not x["conf"]:
        missing.append("還沒有改善證據（財報加速、EPS 上修、催化事件、止跌反轉都沒有）")
    if x["dd_type"] == "UNKNOWN":
        missing.append("下跌原因未判定")
    if x["dd_type"] == "VALUATION_RESET" or (x["val"] is not None and x["val"] < 40):
        missing.append(f"估值分數 {_n(x['val'])} 偏低：還不便宜" if x["val"] is not None else "估值資料不足")
    if x["acc"] is not None and x["acc"] < 40:
        missing.append(f"財報加速分數 {_n(x['acc'])} 偏弱")
    if x["speculative"]:
        missing.append("獲利或營收規模不足以支撐估值")
    return {
        "missing_evidence": missing or ["證據大致齊全，但加碼評分仍未達分批門檻"],
        "watch": ["回撤分類（每日更新）", "確認訊號：財報加速／EPS 預估上修／催化／止跌反轉", "估值分數", "衝擊損害分數"],
        "start_condition": "回撤分類為暫時衝擊或基本面折價、出現至少 1 個確認訊號，且加碼評分 ≥ 55 → 可小量分批",
        "exit_condition": "回撤分類轉為基本面受損、衝擊損害 ≥ 75，或跌出保留名單且 3 年報酬轉負 → 改列退出候選",
    }


def sell_priority(x: Dict, status: str, weight_pct: float, cap_pct: float, universe: int) -> float:
    """釋出資金的優先順序（越高越先賣）：長期邏輯、基本面惡化、動能、估值、集中度、資料信心。"""
    s = {"broken": 40, "weak": 20, "unknown": 15, "intact": 0}.get(status, 10)
    s += min(20.0, x["damage"] / 5)
    if x["rank"]:
        s += 15 * min(1.0, x["rank"] / max(universe, 1))
    if x["val"] is not None:
        s += (100 - x["val"]) / 10
    if weight_pct > cap_pct:
        s += 10
    if not x["has_f"]:
        s += 5
    return round(s, 1)


# --------------------------------------------------------------------------- 套用
def apply_decisions(rows: List[Dict], report: Dict, fundamentals: Optional[Dict[str, Dict]] = None,
                    total_twd: float = 0.0, fx: float = 32.0, live: bool = True) -> None:
    """就地改寫 evaluate_holdings 的每列：加上 decision 欄位，必要時改 action/today（原值存 baseline_*）。"""
    from src.strategy.momentum import CONCENTRATION_PCT, KEEP_N, TOP_N, is_etf
    syms = [h["symbol"] for h in rows if not is_etf(h["symbol"])]
    fund = fundamentals if fundamentals is not None else load_fundamentals(syms, live=live)
    by = {r["symbol"]: r for m in (report.get("markets") or {}).values() for r in m.get("rows", [])}
    usize = {m: len(v.get("rows", [])) for m, v in (report.get("markets") or {}).items()}
    cap = CONCENTRATION_PCT
    for h in rows:
        sym, m = h["symbol"], h["market"]
        h["baseline_action"], h["baseline_today"] = h["action"], h.get("today")
        wpct = (h["value_twd"] / total_twd * 100) if total_twd else (h.get("weight_pct") or 0)
        if is_etf(sym):
            h["decision"] = {"group": "CORE", "decision": "HOLD", "label": "核心 ETF", "thesis": "intact",
                             "why": "ETF 屬核心部位，不套用個股規則。", "confidence": {"confirmed": ["股價、持股（你輸入）"], "estimated": [], "missing": []}}
            continue
        r, f = by.get(sym), fund.get(sym)
        x = _facts(h, r, f, report)
        status, swhy = thesis_status(x, KEEP_N[m])
        a = add_score(x, wpct, cap)
        dropped = (x["dd52"] is not None and x["dd52"] <= -15) or (x["pnl"] is not None and x["pnl"] <= -10)
        grade, gwhy = add_grade(x, a, status, wpct, cap) if dropped else (None, None)
        conf = {"confirmed": ["股價與動能排名（每日收盤／即時報價）", "股數與成本（你輸入）"],
                "estimated": (["基本面／估值分數（yfinance 即時、非歷史時點、未回測）", "回撤分類（規則判定）"] if x["has_f"] else [])
                + (["分批價位（ATR／波動推算）"] if grade in ("左側布局可買", "低檔分批") else []),
                "missing": ((["基本面資料載入中（約 1 分鐘後重新整理）"] if sym in PENDING else ["基本面資料（抓不到）"])
                            if not x["has_f"] else []) + ALWAYS_MISSING}
        d = {"thesis": status, "thesis_why": swhy, "add_score": a, "add_grade": grade, "add_grade_why": gwhy,
             "decline_reason": CAUSE_LABEL.get(x["dd_type"], "資料不足") if dropped else None,
             "decline_detail": x["dd_why"], "valuation_view": _valuation_text(x),
             "sell_priority": sell_priority(x, status, wpct, cap, usize.get(m, 100)),
             "weight_pct": round(wpct, 1), "confidence": conf, "lots": h.get("lots", 1)}
        act = h["action"]
        rank_txt = f"排名第 {x['rank']} 名" if x["rank"] else "排名不明"

        # 2. 集中度 → 部分減碼
        if act == "減碼":
            d.update(group="CORE" if status in ("intact", "weak", "unknown") else "EXIT", decision="TRIM",
                     why=f"部位 {wpct:.1f}% 超過單檔上限 {cap:.0f}%：部分減碼控制風險，不是清倉。" +
                     ("長期邏輯仍在 → 減到上限後續抱。" if status != "broken" else f"另外長期邏輯已破壞（{swhy}）→ 列退出候選。"))
        # 3. 長期邏輯破壞＋動能也弱 → 全數退出
        elif status == "broken" and (x["rank"] or 0) > TOP_N[m]:
            if act != "賣出換股":
                h["action"] = "賣出換股"
                h["today"], h["today_reason"] = "下一步換股", f"長期邏輯已破壞（{swhy}），{rank_txt}、動能也不在前段：列入下一次換股。"
            d.update(group="EXIT", decision="EXIT_ALL", why=f"長期邏輯已破壞：{swhy}；{rank_txt}。帳面損益不影響這個判斷。")
        # 4. 排名跌出保留名單 → 複查
        elif act == "賣出換股":
            if strong_core(x, status) and dropped and grade in ("左側布局可買", "低檔分批"):
                h["action"] = "續抱"
                review = f"{rank_txt}、跌出保留名單 → 觸發複查：基本面通過（{swhy}），且價格已回落、加碼評分 {_n(a['score'])}"
                h["today"], h["today_reason"] = "不用動", review + f" → 續抱並考慮左側分批（{grade}）。（複查與加碼規則未經回測）"
                h["reason"] = h["today_reason"]
                d.update(group="LEFT_SIDE", decision="HOLD_CONSIDER_ADD", why=h["today_reason"],
                         plan=tranche_plan(x, grade, a, h["value_twd"], total_twd, cap, h.get("price"), fx, m))
                _apply_plan(h, d, grade)
            elif strong_core(x, status):
                h["action"], h["today"] = "續抱", "不用動"
                h["today_reason"] = (f"{rank_txt}、跌出保留名單 → 觸發複查：基本面通過（{swhy}）→ 續抱、停止加碼。"
                                     "若回撤分類轉為基本面受損或品質／成長跌破門檻，就換股。（複查規則未經回測）")
                h["reason"] = h["today_reason"]
                d.update(group="CORE", decision="HOLD_NO_ADD", why=h["today_reason"])
            else:
                why = (f"{rank_txt}、跌出保留名單，複查結果：" +
                       ({"weak": f"基本面偏弱（{swhy}）", "unknown": "基本面資料不足，沿用回測規則",
                         "intact": f"基本面未達續抱門檻（需品質 ≥ 60、成長 ≥ 50，下跌屬暫時衝擊／折價；目前品質 {x['q'] or 0:.0f}、成長 {x['g'] or 0:.0f}）"}
                        .get(status, swhy))
                       + " → 換股（回測：跌出名單就換 美 52.3% / 台 49.5%）。")
                d.update(group="EXIT", decision="EXIT_ALL", why=why)
        # 5. 價格回落 → 獨立加碼評估
        elif dropped and grade in ("左側布局可買", "低檔分批"):
            plan = tranche_plan(x, grade, a, h["value_twd"], total_twd, cap, h.get("price"), fx, m)
            d.update(group="LEFT_SIDE", decision="HOLD_CONSIDER_ADD", plan=plan,
                     why=f"{grade}（加碼評分 {_n(a['score'])}）：{gwhy} 下跌原因：{d['decline_reason']}。")
            _apply_plan(h, d, grade)
        elif dropped and grade == "等待確認":
            if act in ("低檔加碼", "加碼"):
                h["action"], h["today"] = "續抱", "不用動"
            h["today_reason"] = f"等待確認（加碼評分 {_n(a['score'])}）：{gwhy}"
            d.update(group="WAIT", decision="HOLD_NO_ADD", why=h["today_reason"], wait=wait_detail(x, a))
        elif dropped and grade in ("不宜加碼", "資料不足，不能判定"):
            if act in ("低檔加碼", "加碼"):
                h["action"], h["today"] = "續抱", "不用動"
            grp = "EXIT" if status == "broken" else ("WAIT" if status in ("weak", "unknown") else "CORE")
            h["today_reason"] = f"{grade}：{gwhy}" + ("帳面虧損不代表便宜。" if (x["pnl"] or 0) < 0 else "")
            d.update(group=grp, decision="HOLD_NO_ADD", why=h["today_reason"],
                     **({"wait": wait_detail(x, a)} if grp == "WAIT" else {}))
        # 6. 其餘
        else:
            grp = "WAIT" if status in ("weak",) else "CORE"
            why = ({"intact": f"長期邏輯仍在（{swhy}）、價格沒有明顯回落 → 續抱；不是低價，不加碼。",
                    "weak": f"基本面偏弱（{swhy}）→ 續抱但停止加碼，等證據。",
                    "unknown": "基本面資料不足，不能判定 → 沿用原規則（續抱）。",
                    "broken": f"長期邏輯有疑慮（{swhy}），但動能仍在前段 → 續抱、停止加碼，動能轉弱就換。"}[status])
            if act == "加碼" and status != "intact":
                h["action"], h["today"], h["today_reason"] = "續抱", "不用動", why
            d.update(group=grp, decision="HOLD" if act == "加碼" and status == "intact" else "HOLD_NO_ADD", why=why,
                     **({"wait": wait_detail(x, a)} if grp == "WAIT" else {}))
        d["label"] = GROUP_LABEL[d["group"]]
        d["decision_label"] = DECISION_LABEL[d["decision"]]
        h["decision"] = d


def _apply_plan(h: Dict, d: Dict, grade: str) -> None:
    """有分批方案時：今天在進場區 → 第 1 筆；不在 → 價位提醒。"""
    plan = d.get("plan") or {}
    if not plan.get("tranches"):
        return
    t1 = plan["tranches"][0]
    h["action"] = "低檔加碼"
    if plan.get("first_tranche_now"):
        h["today"], h["today_reason"] = "左側分批第 1 筆", f"{grade}：第 1 筆約 NT${t1['amount_twd']:,.0f}（{t1['pct']}%），{t1['trigger']}。"
    else:
        h["today"], h["today_reason"] = "等第 1 筆價位", f"{grade}：第 1 筆約 NT${t1['amount_twd']:,.0f}，{t1['trigger']}。"
    h["add_twd"] = t1["amount_twd"]


def _valuation_text(x: Dict) -> str:
    if x["val"] is None:
        return "資料不足"
    lvl = "偏便宜" if x["val"] >= 65 else ("合理" if x["val"] >= 45 else "偏貴")
    return f"估值分數 {_n(x['val'])}（{lvl}，估值類型 {x['val_class'] or '—'}）；無歷史百分位與同業比較"


# --------------------------------------------------------------------------- 報告
def cash_scenarios(rows: List[Dict], total_twd: float, cash_twd: float, fx: float,
                   targets: Iterable[float] = (300_000, 500_000, 1_000_000)) -> List[Dict]:
    """回收現金方案：先賣退出候選，再減碼集中部位，再部分減碼等待確認，最後才動核心；左側布局候選最後。"""
    from src.strategy.momentum import CONCENTRATION_PCT
    order = {"EXIT": 0, "WAIT": 2, "CORE": 3, "LEFT_SIDE": 4}
    cands = []
    for h in rows:
        d = h.get("decision") or {}
        if d.get("label") == "核心 ETF" or not h.get("value_twd"):
            continue
        g = d.get("group", "CORE")
        pr = d.get("sell_priority", 0)
        if d.get("decision") == "TRIM":
            trim = h.get("trim_twd") or max(0.0, h["value_twd"] - total_twd * CONCENTRATION_PCT / 100)
            cands.append((1, -pr, h, trim, "部分減碼：超過單檔上限"))
            if g == "EXIT":
                cands.append((0, -pr, h, h["value_twd"] - trim, "退出候選：" + d.get("why", "")[:40]))
        elif g == "EXIT":
            cands.append((0, -pr, h, h["value_twd"], "退出候選：" + d.get("why", "")[:40]))
        elif g == "WAIT":
            cands.append((2, -pr, h, h["value_twd"] * 0.5, "等待確認、信心較低：先減一半"))
        elif g == "CORE":
            cands.append((3, -pr, h, h["value_twd"] * 0.3, "核心持有：只在前面不夠時減 30%"))
        else:
            cands.append((4, -pr, h, h["value_twd"] * 0.3, "左側布局候選：最後才動"))
    cands.sort(key=lambda c: (c[0], c[1]))
    out = []
    for tgt in targets:
        left, picks, sold = float(tgt), [], {}
        for _, _, h, amt, why in cands:
            if left < 1:
                break
            avail = amt - sold.get(h["symbol"], 0.0)
            take = min(avail, left)
            if take < 1_000:
                continue
            sold[h["symbol"]] = sold.get(h["symbol"], 0.0) + take
            px = (h.get("price") or 0) * (1.0 if h["market"] == "tw" else fx)
            n = min(int(math.ceil(take / px)), int(h["shares"])) if px > 0 else None
            picks.append({"symbol": h["symbol"], "amount_twd": round(take), "shares": n,
                          "full_exit": abs(take - h["value_twd"]) < 1, "reason": why})
            left -= take
        stock_after = sum(h["value_twd"] for h in rows) - sum(sold.values())
        tot = total_twd
        top_after = max(((h["value_twd"] - sold.get(h["symbol"], 0.0)) / tot * 100 for h in rows), default=0) if tot else 0
        out.append({"target_twd": round(tgt), "raised_twd": round(tgt - max(left, 0)), "enough": left < 1, "sells": picks,
                    "after": {"stock_pct": round(stock_after / tot * 100, 1) if tot else None,
                              "cash_pct": round((cash_twd + sum(sold.values())) / tot * 100, 1) if tot else None,
                              "max_single_pct": round(top_after, 1)}})
    return out


def build_decision_report(ev: Dict, plan: Optional[Dict], cash_twd: float, fx: float,
                          baseline: Optional[Dict] = None) -> Dict:
    """每日報告用：今日整體操作、四分組、左側布局清單、賣出與資金調度、今日不動作理由、資料可信度、前後對照。"""
    rows = [h for h in ev.get("holdings", []) if h.get("decision")]
    total = (ev.get("total_twd") or 0) + (cash_twd or 0)
    groups = {g: [] for g in GROUP_LABEL}
    for h in rows:
        groups[h["decision"]["group"]].append(h["symbol"])
    sell_today = [h for h in rows if str(h.get("today", "")).startswith(("今天減碼", "今天賣出"))]
    buy_today = [h for h in rows if h.get("today") in ("左側分批第 1 筆", "今天可加碼", "今天加碼")]
    plan_buys = [s for s in (plan or {}).get("steps", []) if s["kind"] == "buy" and s["when"].startswith("今天")]
    left = [h for h in rows if h["decision"]["group"] == "LEFT_SIDE"]
    exits = sorted([h for h in rows if h["decision"]["group"] == "EXIT" or h["decision"].get("decision") == "TRIM"],
                   key=lambda h: -h["decision"].get("sell_priority", 0))
    reserve = None
    if plan and plan.get("target"):
        reserve = (plan["target"].get("cash") or {}).get("twd")
    full = reserve is not None and cash_twd <= reserve
    top = sorted(rows, key=lambda h: -(h.get("value_twd") or 0))
    max_w = top[0]["decision"].get("weight_pct") if top else None
    if sell_today or buy_today or plan_buys:
        verdict = "今天要操作：" + "、".join(
            ([f"減碼／賣出 {', '.join(h['symbol'] for h in sell_today)}"] if sell_today else [])
            + ([f"買進 {', '.join(sorted({*(h['symbol'] for h in buy_today), *(s['symbol'] for s in plan_buys)}))}"] if (buy_today or plan_buys) else []))
    else:
        verdict = "今天維持不動"
    realloc = None
    funded_today = [s_ for s_ in plan_buys if any(h["symbol"] == s_["symbol"] for h in left)]
    if funded_today and sell_today:
        realloc = (f"今天減碼 {', '.join(h['symbol'] for h in sell_today)} 的錢，先用在 "
                   f"{', '.join(s_['symbol'] for s_ in funded_today)} 的左側第 1 筆；其餘依配置建議投入。")
    elif full and left:
        src = [h for h in exits if h["decision"]["group"] == "EXIT"][:3]
        if src:
            realloc = (f"現金在保留水位內：要布局 {', '.join(h['symbol'] for h in left[:3])}，資金應先從退出候選 "
                       f"{', '.join(h['symbol'] for h in src)} 釋出（低信心 → 高信心）。注意：回測顯示「賣最弱持股來加碼」"
                       "在純動能策略中報酬較差，這裡的移轉依據的是基本面判斷，未經回測。")
        else:
            realloc = "現金在保留水位內、也沒有退出候選：左側布局等新資金，不硬賣其他持股。"
    overall = {
        "verdict": verdict,
        "cash": {"cash_twd": round(cash_twd or 0), "cash_pct": round((cash_twd or 0) / total * 100, 1) if total else None,
                 "reserve_twd": reserve, "fully_invested": full},
        "concentration": {"max_symbol": top[0]["symbol"] if top else None, "max_pct": max_w,
                          "over_cap": [h["symbol"] for h in rows if h["decision"].get("decision") == "TRIM"]},
        "left_side_count": len(left),
        "reallocation": realloc,
    }
    left_list = []
    for h in left:
        d = h["decision"]
        left_list.append({"symbol": h["symbol"], "value_twd": h["value_twd"], "pnl_pct": h.get("pnl_pct"),
                          "decline_reason": d.get("decline_reason"), "decline_detail": d.get("decline_detail"),
                          "thesis": d.get("thesis_why"), "valuation": d.get("valuation_view"),
                          "add_score": d["add_score"], "grade": d.get("add_grade"), "plan": d.get("plan")})
    sell_list = []
    for h in exits:
        d = h["decision"]
        amt = h.get("trim_twd") if d.get("decision") == "TRIM" else h["value_twd"]
        sell_list.append({"symbol": h["symbol"], "type": d.get("decision_label"), "amount_twd": round(amt or 0),
                          "shares": h.get("trim_shares") if d.get("decision") == "TRIM" else int(h["shares"]),
                          "when": (f"排隊第 {d['queued']} 順位（現在不用動）" if d.get("queued") else h.get("today")),
                          "reason": d.get("why"),
                          "use_of_funds": "左側布局候選的下一筆" if left else "動能前段名單（策略精選）",
                          "why_this_one": f"釋出優先分數 {d.get('sell_priority')}（長期邏輯、基本面惡化、動能、估值、集中度、資料信心綜合）"})
    no_action = []
    for h in top[:8]:
        d = h["decision"]
        if h in sell_today or h in buy_today:
            continue
        no_action.append({"symbol": h["symbol"], "group": d.get("label"), "decision": d.get("decision_label"),
                          "why": h.get("today_reason") or d.get("why")})
    base_by = {h["symbol"]: h for h in (baseline or {}).get("holdings", [])}
    diff = []
    for h in rows:
        b = base_by.get(h["symbol"]) or {"action": h.get("baseline_action"), "today": h.get("baseline_today")}
        if b.get("action") != h["action"] or b.get("today") != h.get("today"):
            diff.append({"symbol": h["symbol"], "before": b.get("action"), "before_today": b.get("today"),
                         "after": h["action"], "after_today": h.get("today"), "why": h["decision"].get("why")})
    return {
        "overall": overall,
        "groups": {GROUP_LABEL[g]: v for g, v in groups.items()},
        "left_side": left_list,
        "sells": sell_list,
        "no_action": no_action,
        "cash_scenarios": cash_scenarios(rows, total, cash_twd or 0, fx),
        "baseline_diff": diff,
        "data_confidence": {h["symbol"]: h["decision"].get("confidence") for h in rows},
        "notes": [NOT_BACKTESTED,
                  "原規則（回測基準）保留在每檔的 baseline_action；「前後對照」列出被本層改動的持股。",
                  "帳面虧損不單獨構成賣出理由；帳面獲利也不單獨構成續抱理由。"],
    }
