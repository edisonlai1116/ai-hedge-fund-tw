"""把所有模組串起來：對一批股票做 cross-sectional ranking，輸出每檔的完整評分卡。

用法：
  from src.ranking.engine import rank_stocks
  out = rank_stocks(["AVGO", "VST", ...], holdings={"VST": {"shares": 65, "cost": 153}})

即時模式（asof=None）：價格模組 + 基本面 + 新聞催化 + 大跌診斷 + 市場狀態。
回測/歷史模式（asof=某日）：只用 point-in-time 的價格模組；基本面/新聞標記 not_point_in_time 並排除。
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ranking import features as fx
from src.ranking import scoring as sc
from src.ranking.regime import classify_regime
from src.ranking.shocks import detect_shocks, fundamental_damage, shock_verdict
from src.ranking.themes import THEMES, TW_THEMES, industry_benchmark, theme_exposures, theme_of

US_BENCHMARKS = ["SPY", "QQQ", "SMH", "XLK", "XLU", "XLE", "XLI", "XLC", "XLY", "IGV"]
TW_BENCHMARKS = ["0050.TW"]

MODULES = {   # 六個量化模組（UI 顯示用名稱）→ 對應的分數欄位
    "估值與品質": ["valuation", "quality"],
    "成長與財報": ["growth", "earnings_acceleration"],
    "技術與動能": ["technical", "relative_strength", "overextension"],
    "情緒與籌碼": ["positioning"],
    "新聞與催化": ["catalyst"],
    "AI 產業與總經": ["ai_exposure", "industry_momentum", "regime"],
}


_DL_CACHE: Dict[tuple, Dict[str, pd.DataFrame]] = {}


def _download(symbols: List[str], period: str) -> Dict[str, pd.DataFrame]:
    """同一程序內快取下載結果（回測逐日呼叫時只切片、不重抓）。"""
    from src.strategy.momentum import download_closes
    key = (tuple(sorted(set(symbols))), period)
    if key not in _DL_CACHE:
        _DL_CACHE[key] = download_closes(list(key[0]), period=period)
    return {k: v.copy() for k, v in _DL_CACHE[key].items()}


def _slice(df: pd.DataFrame, asof: Optional[pd.Timestamp]) -> pd.DataFrame:
    if asof is None:
        return df
    i = pd.to_datetime(df.index)
    i = i.tz_localize(None) if i.tz is not None else i
    return df[i <= asof]


def _is_tw(s: str) -> bool:
    return s.upper().endswith((".TW", ".TWO"))


def _to_yf(t: str) -> str:
    t = t.strip().upper()
    return t if "." in t else (f"{t}.TW" if t[:1].isdigit() else t)


def _reference_universe(market: str) -> List[str]:
    if market == "tw":
        return [f"{k}.TWO" if k in ("8299", "6274", "3324", "5274", "3529", "6488", "3105") else f"{k}.TW" for k in TW_THEMES]
    return list(THEMES)


def _positioning(info: Dict) -> Dict:
    """情緒與籌碼（即時）：放空比例、機構持股、分析師共識的變化不在免費資料內 → 只用可得欄位。"""
    spf = info.get("shortPercentOfFloat")
    inst = info.get("heldPercentInstitutions")
    rec = info.get("recommendationMean")   # 1=強力買進 … 5=賣出
    parts = {
        "short_interest": (None if spf is None else max(0.0, min(100.0, 100 - spf / 0.20 * 100)), 0.4),
        "analyst_consensus": (None if rec is None else max(0.0, min(100.0, (4.0 - rec) / 2.5 * 100)), 0.6),
    }
    used = {k: v for k, v in parts.items() if v[0] is not None}
    if not used:
        return {"score": None, "metrics": {}}
    s = sum(v * w for v, w in used.values()) / sum(w for _, w in used.values())
    return {"score": round(s, 1), "metrics": {"short_pct_float": spf, "institutional_pct": inst, "recommendation_mean": rec},
            "note": "以放空比例與分析師共識為代理；內部人交易需 financialdatasets API（LLM 流程中提供）"}


def rank_stocks(tickers: List[str], holdings: Optional[Dict[str, Dict]] = None, asof: Optional[str] = None,
                live_fundamentals: bool = True, live_news: bool = True, calibration: Optional[Dict] = None,
                universe_ranks: Optional[Dict[str, tuple]] = None) -> Dict:
    """universe_ranks：{代號: (策略股票池名次, 池大小)}（strategy.json）——Investment Quality Tier 用；
    沒提供時以本批參考池的動能百分位換算。"""
    holdings = {_to_yf(k): v for k, v in (holdings or {}).items()}
    batch = [_to_yf(t) for t in tickers]
    asof_ts = pd.Timestamp(asof) if asof else None
    point_in_time_only = asof is not None
    if point_in_time_only:
        live_fundamentals = live_news = False

    markets = {"us": [s for s in batch if not _is_tw(s)], "tw": [s for s in batch if _is_tw(s)]}
    period = "4y" if asof is None else "10y"
    from src.ranking.regime import TICKERS as REGIME_TICKERS
    rpm = _download(list(REGIME_TICKERS.values()), period)
    regime_inputs = {k: rpm[v]["Close"] for k, v in REGIME_TICKERS.items() if v in rpm}
    regime = classify_regime(regime_inputs, asof_ts)
    profile = regime["profile"]

    rows: List[Dict] = []
    for market, syms in markets.items():
        if not syms:
            continue
        ref = list(dict.fromkeys(_reference_universe(market) + syms))
        benches = US_BENCHMARKS if market == "us" else TW_BENCHMARKS
        pm = {k: _slice(v, asof_ts) for k, v in _download(ref + benches, period).items()}
        pm = {k: v for k, v in pm.items() if len(v) > 30}
        bench_close = {b: pm.pop(b)["Close"] for b in benches if b in pm}
        if market == "tw" and "0050.TW" in bench_close:
            bench_close["SPY"] = bench_close["QQQ"] = bench_close["0050.TW"]
        themes = {s: (theme_of(s).theme if theme_of(s) else None) for s in pm}
        inds = {s: industry_benchmark(s) for s in pm}
        F = fx.compute_features(pm, bench_close, themes, inds)
        panels = {
            "momentum_rank": fx.pct_rank(F["sharpe_126"]),
            "relative_strength": fx.relative_strength_score(F),
            "technical": fx.technical_score(F),
            "industry_momentum": fx.industry_momentum_score(F),
            "risk": fx.risk_score(F),
            "overextension": fx.overextension_score(F),
        }
        last = {k: fx.last_row(v, asof_ts) for k, v in panels.items()}
        lastF = {k: fx.last_row(v, asof_ts) for k, v in F.items()}

        def build(sym: str) -> Optional[Dict]:
            if sym not in pm:
                return {"ticker": sym, "status": "WAIT", "error": "抓不到報價"}
            return _build_row(sym, pm[sym], F, last, lastF, bench_close, profile, holdings.get(sym),
                              live_fundamentals, live_news, calibration, (universe_ranks or {}).get(sym))

        with ThreadPoolExecutor(max_workers=6) as ex:
            rows += [r for r in ex.map(build, syms) if r]

    good = [r for r in rows if r.get("opportunity") is not None]
    good.sort(key=lambda r: r["opportunity"], reverse=True)
    for i, r in enumerate(good, 1):
        r["batch_rank"] = i
    for r in good:   # Better than / Worse than：同批次內機會分數相鄰者
        better = [o["ticker"] for o in good if o["opportunity"] < r["opportunity"] - 5][:3]
        worse = [o["ticker"] for o in good if o["opportunity"] > r["opportunity"] + 5][-3:]
        r["rotation_view"] = {"better_than": better, "worse_than": worse}
    rotations = sc.rotation_suggestions(good, calibration)
    # 組合層級：主題曝險只影響部位大小，不回寫任何個股分數
    exposures = {r["ticker"]: r.get("theme_exposure") or {"Other": 1.0} for r in good}
    sizes = sc.position_sizes(good, profile, calibration, exposures=exposures)
    for r in good:
        r["target_weight"] = sizes.get(r["ticker"], 0.0)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof or "latest",
        "mode": "point_in_time_price_only" if point_in_time_only else "live",
        "regime": {"regime": regime["regime"], "metrics": regime["metrics"], "profile": profile},
        "ranking": good + [r for r in rows if r.get("opportunity") is None],
        "rotations": rotations,
        "target_weights": sizes,
        "theme_exposure": {k: round(v, 4) for k, v in sc.theme_exposure(sizes, exposures).items()},
        "theme_cap": sc.THEME_CAP,
        "weights_used": sc.regime_weights(profile),
        "notes": [
            "Opportunity Score = 股票本身值不值得投資（動能排名 + 品質/成長/財報/估值/AI/催化），不含進場時機與組合資訊。",
            "Entry Score / entry_condition = 現在這個價格適不適合進場；BUY_NOW 一定要 entry_condition 成立。",
            "Overextension 依歷史百分位分層（Normal/Warm/Extended/Overheated/Extreme）；好公司但過熱 → GOOD_BUT_OVEREXTENDED，不追。",
            "大跌先做回撤分類，只有 FUNDAMENTAL_DISCOUNT / TEMPORARY_SHOCK 允許大跌 → BUY_NOW。",
            "主題曝險上限只縮小部位，不改個股分數。",
            "基本面、新聞只有即時資料，沒有 point-in-time 歷史 → 不參與回測；回測只驗證價格模組。",
            "AI 曝險分數為專家先驗，不參與回測。",
            "權重依驗證結果：已驗證的動能排名占 61%；其餘模組未經回測驗證，每檔顯示 validated_share。",
            "過熱不扣 Opportunity 分數，改為進場規則：Extended（≥ 50）→ BUY_ON_PULLBACK、Extreme（≥ 85）→ WAIT。"
            "注意：歷史上過熱股之後的報酬並不較差（動能延續），不追高是紀律／風險控制，不是額外報酬來源。",
            "市場狀態只調整風險上限（高 beta 單檔上限），不調整總曝險：回測顯示依狀態減碼會降低 Sharpe。",
        ],
    }


def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def decision_pipeline(ohlcv: pd.DataFrame, scores: Dict, profile: Optional[Dict], dd52: Optional[float], ret_3y: Optional[float],
                      dist_ma50: Optional[float], shock_view: Optional[Dict], fund_metrics: Optional[Dict],
                      days_to_earnings: Optional[int], held: bool, ret_20: Optional[float] = None,
                      universe_rank: Optional[tuple] = None) -> Dict:
    """確定性決策（純函式，不連網）：A Opportunity → B Entry → C Overextension（已在 scores）→ D Risk → 回撤分類 → E Action。
    會在 scores 補上 low_entry、price_risk、fundamental_uncertainty 並把 risk 改為合成風險。正式流程與測試共用。"""
    weights = sc.regime_weights(profile)
    comp = {k: v for k, v in scores.items() if k in sc.BASE_WEIGHTS}
    opp = sc.opportunity_score(comp, None, weights)
    low_score = sc.low_entry_score(dd52, ret_3y, None)
    scores["low_entry"] = low_score
    is_low = bool(dd52 is not None and dd52 <= sc.LOW_ENTRY_DD and ret_3y is not None and ret_3y > 0)
    lt_broken = ret_3y is not None and ret_3y <= 0
    damage = (shock_view or {}).get("fundamental_damage_score")
    oversold = bool(shock_view and shock_view.get("verdict") == "POTENTIAL_OVERSOLD")
    zones = sc.entry_zones(ohlcv, scores.get("valuation"), days_to_earnings)
    close_s = ohlcv["Close"].dropna()
    low_line = (round(float(close_s.tail(252).max()) * (1 + sc.LOW_ENTRY_DD), 2)
                if (ret_3y is not None and ret_3y > 0 and len(close_s) >= 200) else None)
    fu = sc.fundamental_uncertainty(fund_metrics)
    dclass = sc.classify_drawdown(dd52, scores.get("quality"), scores.get("growth"), scores.get("earnings_acceleration"),
                                  scores.get("valuation"), damage, fu, (fund_metrics or {}).get("eps_revision_90d"),
                                  ((shock_view or {}).get("shock") or {}).get("causes"))
    # 低檔區、且回撤屬可買類：Buy1 = 已驗證低檔區（現價 + 0.5 ATR，上限為 52 週高點 −30% 線）
    px, atr = zones.get("price"), zones.get("atr")
    if (is_low and dclass["type"] in sc.BUYABLE_DRAWDOWNS and low_line and px and atr and zones.get("zone_1")
            and px <= low_line and zones["zone_1"]["high"] < px):
        hi = round(min(low_line, px + 0.5 * atr), 2)
        zones["zone_1"] = {"low": zones["zone_1"]["low"], "high": hi,
                           "basis": f"已驗證低檔區（52 週高點 −30% 線 {low_line} 以下）", "pct_from_price": round((hi / px - 1) * 100, 1)}
    cond = sc.entry_condition(zones.get("price"), zones, scores.get("overextension"), low_line)
    zones["low_entry_line"] = low_line
    zones["condition"] = cond
    e_score = sc.entry_score(cond, low_score, scores.get("valuation"), dist_ma50)
    scores["price_risk"] = scores.get("risk")
    scores["fundamental_uncertainty"] = fu["score"] if fu else None
    scores["risk"] = sc.combined_risk(scores["price_risk"], fu)
    # A. Price Location / B. Investment Quality Tier / 獨立 thesis 證據（與跌多深無關）
    location = sc.price_location(dd52)
    tier = (sc.quality_tier(universe_rank[0], universe_rank[1]) if universe_rank
            else sc.quality_tier(percentile=scores.get("momentum_rank")))
    conf = sc.thesis_confirmations(scores.get("earnings_acceleration"), (fund_metrics or {}).get("eps_revision_90d"),
                                   scores.get("catalyst"), ret_20, dist_ma50, scores.get("growth"))
    args = dict(opp=opp["score"], overext=scores.get("overextension"), quality=scores.get("quality"), risk=scores["risk"],
                valuation=scores.get("valuation"), relative_strength=scores.get("relative_strength"), damage=damage,
                oversold=oversold, entry_ok=cond["ok"], low_entry=is_low, long_term_broken=lt_broken,
                growth=scores.get("growth"), drawdown_type=dclass["type"], tier=tier, confirmations=conf)
    st = sc.decide_action(held=held, **args)
    new_money = sc.decide_action(held=False, **args) if held else st
    for a in {id(st): st, id(new_money): new_money}.values():
        target = {"ABOVE_ENTRY": "zone_1", "COOLING_REQUIRED": "zone_2", "CONFIRMATION_REQUIRED": "zone_2"}.get(a.get("pullback_reason"))
        if a["status"] == "BUY_ON_PULLBACK" and target and zones.get(target):
            z = zones[target]
            a["why"] += f"（{'Buy1' if target == 'zone_1' else 'Buy2'} {z['low']}–{z['high']}）"
    return {"price_location": location, "quality_tier": tier, "confirmations": conf,
            "opp": opp, "weights": weights, "zones": zones, "cond": cond, "entry_score": e_score, "fu": fu, "drawdown": dclass,
            "action": st, "new_money": new_money, "low_score": low_score, "is_low": is_low, "lt_broken": lt_broken}


def _build_row(sym, ohlcv, F, last, lastF, bench_close, profile, holding, live_fund, live_news, calibration,
               universe_rank: Optional[tuple] = None) -> Dict:
    th = theme_of(sym)
    g = lambda name: _num(lastF[name].get(sym)) if name in lastF else None   # noqa: E731
    scores = {
        "momentum_rank": _num(last["momentum_rank"].get(sym)),
        "relative_strength": _num(last["relative_strength"].get(sym)),
        "technical": _num(last["technical"].get(sym)),
        "industry_momentum": _num(last["industry_momentum"].get(sym)),
        "risk": _num(last["risk"].get(sym)),
        "overextension": _num(last["overextension"].get(sym)),
        "ai_exposure": float(th.ai_exposure) if (th and live_fund) else None,
    }
    fundamentals, catalyst, positioning, valuation_class, next_er = {}, None, None, None, None
    fund_metrics: Optional[Dict] = None
    if live_fund:
        try:
            from src.ranking.fundamentals import FundamentalSnapshot
            snap = FundamentalSnapshot(sym)
            q, gr, ea = snap.quality(), snap.growth(), snap.earnings_acceleration()
            val = snap.valuation()
            pos = _positioning(snap.info)
            fundamentals = {"quality": q, "growth": gr, "earnings_acceleration": ea, "valuation": val}
            scores.update({"quality": q["score"], "growth": gr["score"], "earnings_acceleration": ea["score"],
                           "valuation": val["score"], "positioning": pos["score"]})
            positioning = pos
            valuation_class = snap.valuation_class
            ner = snap.next_earnings_date()
            next_er = None if ner is None else int((ner - pd.Timestamp.now(tz=ner.tz)).days)
            company = snap.info.get("shortName") or ""
            inf = snap.info
            fund_metrics = {
                "operating_margin": _num(inf.get("operatingMargins")),
                "fcf_margin": (q.get("metrics") or {}).get("fcf_margin"),
                "revenue": _num(inf.get("totalRevenue")),
                "net_debt_to_ebitda": (q.get("metrics") or {}).get("net_debt_to_ebitda"),
                "eps_revision_90d": (ea.get("metrics") or {}).get("eps_estimate_revision_90d"),
                "days_to_earnings": next_er,
                "sector": inf.get("sector"),
            }
        except Exception as exc:
            fundamentals = {"error": f"{type(exc).__name__}: {exc}"}
            company = ""
    else:
        company = ""
    sec_val = None
    if live_fund and not _is_tw(sym) and os.environ.get("SEC_USER_AGENT"):
        try:   # SEC point-in-time：相對自身 3 年 P/S（資訊用；回測顯示當作篩選條件並未改善報酬）
            from src.data.sec_fundamentals import pit_valuation
            v = pit_valuation(sym, ohlcv["Close"].dropna())
            if v is not None and v["ps"].notna().sum() > 300:
                ps = v["ps"].dropna()
                med = float(ps.tail(756).median())
                sec_val = {"ps": round(float(ps.iloc[-1]), 2), "ps_3y_median": round(med, 2),
                           "relative_to_own_3y": round(float(ps.iloc[-1]) / med, 2) if med > 0 else None,
                           "pe": round(float(v["pe"].dropna().iloc[-1]), 1) if v["pe"].notna().any() else None,
                           "source": "SEC XBRL（依申報日，point-in-time）"}
        except Exception as exc:
            sec_val = {"error": f"{type(exc).__name__}"}
    if live_news:
        try:
            from src.ranking.catalysts import catalyst_report
            bench = bench_close.get("SPY")
            catalyst = catalyst_report(sym.split(".")[0], company, ohlcv["Close"], bench)
            scores["catalyst"] = catalyst.get("score")
        except Exception as exc:
            catalyst = {"score": None, "error": str(exc)}

    # 大跌診斷：與大盤、同主題同日表現比較
    theme_ret_1 = None
    if th:
        peers = [s for s in F["ret_1"].columns if s != sym and theme_of(s) and theme_of(s).theme == th.theme]
        if peers:
            theme_ret_1 = F["ret_1"][peers].mean(axis=1)
    ox_series = fx.overextension_score(F)[sym].dropna()
    ox_before = _num(ox_series.iloc[-7]) if len(ox_series) > 7 else None
    spy_ret_1 = F["spy_ret_1"][sym] if "spy_ret_1" in F else None
    shocks = detect_shocks(F["ret_1"][sym], spy_ret_1, theme_ret_1, g("beta_252"), ox_before)
    shock_view = None
    if shocks:
        worst = min(shocks, key=lambda s: s["return"])
        ea_m = (fundamentals.get("earnings_acceleration") or {}).get("metrics", {}) if live_fund else {}
        close = F["close"][sym].dropna()
        at_shock = close.loc[:pd.Timestamp(worst["date"])]
        since = float(close.iloc[-1] / at_shock.iloc[-1] - 1) if len(at_shock) else None
        dmg = fundamental_damage(worst, (catalyst or {}).get("events") if live_news else None,
                                 ea_m.get("eps_estimate_revision_90d"), ea_m.get("latest_surprise_pct"), since)
        shock_view = {"shock": worst, **dmg, "verdict": shock_verdict(worst, dmg)}

    d = decision_pipeline(ohlcv, scores, profile, dd52=g("drawdown_252"), ret_3y=g("ret_756"), dist_ma50=g("dist_ma50"),
                          shock_view=shock_view, fund_metrics=fund_metrics, days_to_earnings=next_er, held=holding is not None,
                          ret_20=g("ret_20"), universe_rank=universe_rank)
    opp, zones, cond, e_score, fu, dclass, st, new_money = (d[k] for k in ("opp", "zones", "cond", "entry_score", "fu",
                                                                         "drawdown", "action", "new_money"))
    weights, low_score, is_low, lt_broken, held = d["weights"], d["low_score"], d["is_low"], d["lt_broken"], holding is not None
    dd52, r3 = g("drawdown_252"), g("ret_756")
    price = zones.get("price") or _num(ohlcv["Close"].dropna().iloc[-1])
    vol63 = g("vol_63")
    row = {
        "ticker": sym, "company": company, "price": price, "held": held,
        "theme": th.theme if th else None, "sub_theme": th.sub_theme if th else None,
        "valuation_class": valuation_class or (th.valuation_class if th else None),
        "opportunity": opp["score"],
        "opportunity_detail": {**opp, "validated_share": round(sum(weights[k] for k in opp.get("components_used", [])
                                                                    if k in sc.VALIDATED_COMPONENTS) /
                                                                sum(weights[k] for k in opp.get("components_used", [])) , 2)
                                                          if opp.get("components_used") else 0.0},
        "status": st["status"], "flags": st["flags"], "status_why": st["why"], "state": st.get("state"),
        "pullback_reason": st.get("pullback_reason"),
        "new_money_action": {"status": new_money["status"], "why": new_money["why"], "state": new_money.get("state")},
        "entry_score": e_score, "entry_condition": cond,
        "overextension_tier": fx.overextension_tier(scores["overextension"]),
        "drawdown": dclass, "drawdown_type": dclass["type"],
        "price_location": d["price_location"], "quality_tier": d["quality_tier"], "confirmations": d["confirmations"],
        "universe_rank": list(universe_rank) if universe_rank else None,
        "low_price": new_money.get("low_price"),
        "risk_detail": {"price_risk": scores["price_risk"], "fundamental_uncertainty": fu},
        "theme_exposure": theme_exposures(sym, (fund_metrics or {}).get("sector")),
        "scores": {**scores, "earnings": scores.get("earnings_acceleration"), "ai": scores.get("ai_exposure"),
                   "industry": scores.get("industry_momentum")},
        "ai": {"exposure": th.ai_exposure if th else None, "demand_sensitivity": th.ai_demand_sensitivity if th else None,
               "cycle_position": _cycle_position(F, sym, th)},
        "returns": {f"{n}d": g(f"ret_{n}") for n in (5, 20, 60, 120, 252, 756)},
        "low_entry": {"in_zone": is_low, "dd_52w": dd52, "ret_3y": r3, "score": low_score, "long_term_broken": lt_broken},
        "relative": {f"vs_{b}_{n}d": g(f"rel_{b}_{n}") for b in ("SPY", "QQQ", "ind", "theme") for n in (20, 60, 120, 252)},
        "technical_detail": {"dist_ma20": g("dist_ma20"), "dist_ma50": g("dist_ma50"), "dist_52w_high": g("dist_52w_high"),
                             "rsi14": g("rsi14"), "volume_spike": g("vol_spike"), "up_volume_ratio": g("up_volume_ratio"),
                             "breakout": bool(g("breakout")) if g("breakout") is not None else None},
        "vol_63": vol63, "beta": g("beta_252"), "downside_risk": (abs(g("drawdown_252")) if g("drawdown_252") is not None else None),
        "coverage": opp["coverage"],
        "entry": zones, "fundamentals": fundamentals, "catalyst": catalyst, "positioning": positioning,
        "sec_valuation": sec_val,
        "shock": shock_view, "holding": holding,
        "expected_excess_60d": sc.expected_excess_return(opp["score"], calibration),
    }
    row["narrative"] = _narrative(row)
    return row


def _cycle_position(F, sym, th) -> Optional[str]:
    """AI 週期位置（以同主題價格行為判斷）：Early / Accelerating / Mature / Overheated。"""
    if not th or "theme_ret_60" not in F:
        return None
    r60 = _num(fx.last_row(F["theme_ret_60"])[sym]) if sym in F["theme_ret_60"] else None
    r252 = _num(fx.last_row(F["theme_ret_252"])[sym]) if "theme_ret_252" in F and sym in F["theme_ret_252"] else None
    ox = fx.overextension_score(F)
    peers = [s for s in ox.columns if theme_of(s) and theme_of(s).theme == th.theme]
    theme_ox = _num(fx.last_row(ox[peers]).mean()) if peers else None
    if r60 is None:
        return None
    if theme_ox is not None and theme_ox >= 65:
        return "Overheated"
    if r60 > 0 and (r252 is None or r252 < 0.30):
        return "Early"
    if r60 > 0:
        return "Accelerating"
    return "Mature"


LABELS = {"low_entry": "低檔分數", "momentum_rank": "動能排名", "quality": "品質", "growth": "成長", "earnings_acceleration": "財報加速", "valuation": "估值",
          "ai_exposure": "AI 曝險", "industry_momentum": "產業動能", "relative_strength": "相對強度",
          "catalyst": "催化", "technical": "技術"}


def _narrative(r: Dict) -> Dict[str, str]:
    s = r["scores"]
    ranked = sorted([(k, v) for k, v in s.items() if k in LABELS and v is not None], key=lambda x: -x[1])
    strong = [f"{LABELS[k]} {v:.0f}" for k, v in ranked[:3] if v >= 60]
    weak = [f"{LABELS[k]} {v:.0f}" for k, v in ranked[::-1][:3] if v < 45]
    ox, risk = s.get("overextension"), s.get("risk")
    z = r["entry"]
    cat = r.get("catalyst") or {}
    shock = r.get("shock") or {}
    why_buy = ("、".join(strong) + "。" if strong else "沒有特別突出的強項。") + (
        "大跌主因非公司基本面，屬潛在超跌。" if "POTENTIAL_OVERSOLD" in r["flags"] else "")
    le = r.get("low_entry") or {}
    if le.get("in_zone"):
        why_buy = f"長線贏家（3 年 {le['ret_3y']:+.0%}）已自 52 週高點回落 {abs(le['dd_52w']):.0%}，落入低檔布局區。" + why_buy
    sv = r.get("sec_valuation") or {}
    if sv.get("relative_to_own_3y") is not None:
        rel = sv["relative_to_own_3y"]
        why_buy += f" 估值為自身 3 年中位數的 {rel:.2f} 倍（P/S {sv['ps']}）" + ("，相對便宜。" if rel <= 0.85 else "。")
    why_not = []
    if ox is not None and ox >= 60:
        why_not.append(f"短線過熱 {ox:.0f}（RSI {r['technical_detail'].get('rsi14') or 0:.0f}、距 20 日線 {100 * (r['technical_detail'].get('dist_ma20') or 0):.0f}%）")
    if s.get("valuation") is not None and s["valuation"] < 35:
        why_not.append(f"估值分數僅 {s['valuation']:.0f}")
    if risk is not None and risk >= 70:
        why_not.append(f"風險分數 {risk:.0f}（波動/beta 高）")
    if z.get("avoid_above") and z.get("price") and z["price"] > z["avoid_above"]:
        why_not.append(f"現價高於 avoid_above {z['avoid_above']}")
    ea = (r.get("fundamentals") or {}).get("earnings_acceleration") or {}
    invalid = [f"收盤跌破 Zone 3 下緣 {z['zone_3']['low']}" if z.get("zone_3") else None,
               "相對強度跌破 30（跑輸同業與大盤）",
               "營收成長由加速轉為減速且 EPS 預估下修 >5%" if ea.get("score") is not None else None]
    key_risk = "、".join(weak) if weak else ("高波動" if (risk or 0) >= 60 else "AI 族群整體回檔")
    if shock.get("verdict") == "THESIS_AT_RISK":
        key_risk = "近期大跌疑似基本面受損：" + "；".join(shock.get("evidence", []))
    return {
        "why_now": why_buy,
        "why_not_chase": "；".join(why_not) if why_not else "未過熱、可在進場區分批。",
        "key_risk": key_risk,
        "attractive_price": (f"{z['zone_1']['low']}–{z['zone_1']['high']}（{z['zone_1']['basis']}）起開始有吸引力"
                             if z.get("zone_1") else "資料不足"),
        "thesis_invalidation": "；".join(x for x in invalid if x),
        "bull_thesis": "、".join(strong) or "—",
        "bear_thesis": "、".join(weak) or "—",
        "key_catalyst": cat.get("key_catalyst") or "近 30 天無明確事件",
    }


def format_card(r: Dict) -> str:
    s, z, n = r["scores"], r["entry"], r["narrative"]
    f = lambda v: "—" if v is None else f"{v:.0f}"   # noqa: E731
    zone = lambda k: f"{z[k]['low']}–{z[k]['high']}（{z[k]['basis']}）" if z.get(k) else "—"   # noqa: E731
    rv = r.get("rotation_view") or {}
    return "\n".join([
        f"Ticker: {r['ticker']}  ({r.get('theme') or '-'} / {r.get('sub_theme') or '-'}, 估值類別 {r.get('valuation_class') or '-'})",
        f"Current Price: {r['price']}",
        f"Opportunity Score: {f(r['opportunity'])}/100  （批次排名 {r.get('batch_rank', '-')}，資料覆蓋 {int((r.get('coverage') or 0) * 100)}%）",
        f"Action: {r['status']}" + (f"  [{', '.join(r['flags'])}]" if r["flags"] else "") + f" — {r['status_why']}",
        "Scores:",
        f"  Quality: {f(s.get('quality'))}  Growth: {f(s.get('growth'))}  Earnings: {f(s.get('earnings_acceleration'))}  AI: {f(s.get('ai_exposure'))}",
        f"  Industry: {f(s.get('industry_momentum'))}  Valuation: {f(s.get('valuation'))}  Technical: {f(s.get('technical'))}  Catalyst: {f(s.get('catalyst'))}",
        f"  Relative Strength: {f(s.get('relative_strength'))}  Risk: {f(s.get('risk'))}  Overextension: {f(s.get('overextension'))}  "
        f"Momentum rank（已驗證）: {f(s.get('momentum_rank'))}  Low-entry（已驗證）: {f(s.get('low_entry'))}",
        f"  AI cycle: {r['ai'].get('cycle_position') or '-'}",
        "Entry:",
        f"  Zone 1: {zone('zone_1')}",
        f"  Zone 2: {zone('zone_2')}",
        f"  Zone 3: {zone('zone_3')}",
        f"  Avoid Above: {z.get('avoid_above')}" + (f"  註：{'；'.join(z['notes'])}" if z.get("notes") else ""),
        f"為什麼現在值得買？ {n['why_now']}",
        f"為什麼現在不能追？ {n['why_not_chase']}",
        f"最重要的風險？ {n['key_risk']}",
        f"什麼價格開始有吸引力？ {n['attractive_price']}",
        f"什麼事件會讓 thesis 失效？ {n['thesis_invalidation']}",
        f"Bull Thesis: {n['bull_thesis']}",
        f"Bear Thesis: {n['bear_thesis']}",
        f"Key Catalyst: {n['key_catalyst']}",
        f"Key Risk: {n['key_risk']}",
        f"Thesis Invalidation: {n['thesis_invalidation']}",
        "Rotation:",
        f"  Better than: {', '.join(rv.get('better_than') or []) or '—'}",
        f"  Worse than: {', '.join(rv.get('worse_than') or []) or '—'}",
    ] + ([f"Price shock: {r['shock']['shock']['date']} {r['shock']['shock']['return']:+.1%}，原因 {', '.join(r['shock']['shock']['causes'])}，"
          f"基本面受損 {r['shock']['fundamental_damage_score']}/100 → {r['shock'].get('verdict') or '無'}"] if r.get("shock") else []))
