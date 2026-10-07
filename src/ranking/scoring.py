"""決策流程（2026-10-08 Decision Logic Consistency Audit 後）——五個概念分開，最後才合成動作：

  A. Opportunity Score（股票本身值不值得投資/研究）：已驗證的 6 個月動能排名 + 品質、成長、財報加速、估值、AI、催化。
     不含進場時機（低檔深度、距支撐距離），不含任何組合資訊（持股、主題曝險）。
  B. Entry Score / Entry Condition（現在這個價格適不適合進場）：距 Buy1 幾個 ATR、已驗證的低檔深度、現價估值、距 50 日線。
     entry_condition() 明確回答「現價是否可立即進場」與理由。
  C. Overextension（短線過熱，features.overextension_score，歷史百分位校準，分 Normal/Warm/Extended/Overheated/Extreme）。
  D. Risk（價格風險 + 基本面不確定性：虧損、負 FCF、營收小、高槓桿、財報在即）。
  E. Final Action：decide_action(A, B, C, D, 回撤分類, 基本面受損, 持有與否)。
  F. Position Sizing（組合層級）：position_sizes() 依預期報酬/風險配置，再套單檔、高 beta、主題曝險上限。
     主題曝險只縮小部位，不改任何個股分數。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 權重依 2026-10-06 point-in-time 驗證（src/ranking/validate.py）決定：
#   * 價格面只有「6 個月風險調整動能排名」(momentum_rank) 在美/台股、樣本內外都穩定勝出；
#     多模組價格綜合分數反而較差 → 價格面以 momentum_rank 為核心。
#   * 產業動能：單獨看「看多 → 20 日跑贏大盤」57.7%（95% CI 56.1~59.2），但加進排名後組合報酬反而下降
#     （美股樣本內 40.2%→35.0%、樣本外 47.9%→40.4%）→ 權重 0，只當診斷資訊顯示。
#   * 相對強度 / 技術 / 低風險：消融測試未增加 alpha → 權重 0，只當診斷資訊顯示。
#   * 基本面、財報加速、估值、催化、AI 曝險：沒有 point-in-time 歷史無法回測 → 保留但權重有限，
#     並在輸出中標記「未經回測驗證」。
# 2026-10-08：Opportunity 改為「公司/股票本身」——不再混入低檔深度（那是進場時機，移到 Entry Score）。
# 價格面仍以已驗證的動能排名為核心；只有價格資料時（回測、台股缺財報）就等於動能排名，回測結果不變。
BASE_WEIGHTS = {
    "momentum_rank": 0.32,          # 已驗證（唯一在組合層級穩定增加 alpha 的價格因子）
    "quality": 0.15,                # 未驗證
    "growth": 0.13,                 # 未驗證
    "earnings_acceleration": 0.13,  # 未驗證
    "valuation": 0.13,              # 未驗證
    "ai_exposure": 0.08,            # 未驗證（專家先驗）
    "catalyst": 0.06,               # 未驗證
}
OPPORTUNITY_WEIGHTS = BASE_WEIGHTS
VALIDATED_COMPONENTS = ("momentum_rank",)

# 長線低檔布局（使用者風格：低檔為主 70%、動能 30%）
LOW_ENTRY_DD = -0.30
ALLOCATION = {"lowentry": 0.70, "momentum": 0.30}


def low_entry_score(dd_52w: Optional[float], ret_3y: Optional[float], valuation: Optional[float] = None) -> Optional[float]:
    """低檔布局分數 0~100：長線贏家（3 年報酬 > 0）距 52 週高點越深分數越高（-30% → 80、-50% → 100）；
    非長線贏家打三折。估值（未經回測驗證）只做 ±10 分微調。"""
    if dd_52w is None:
        return None
    depth = min(100.0, max(0.0, (-dd_52w) / 0.30 * 80 if dd_52w >= -0.30 else 80 + (-dd_52w - 0.30) / 0.20 * 20))
    if ret_3y is None or ret_3y <= 0:
        depth *= 0.3
    if valuation is not None:
        depth = depth * 0.9 + valuation * 0.1
    return round(max(0.0, min(100.0, depth)), 1)
DIAGNOSTIC_ONLY = ("relative_strength", "technical", "low_risk", "industry_momentum")
# 過熱「不扣排名分數」（回測顯示扣分會降低報酬）；改由進場規則處理：
# 排名夠前但過熱 ≥ OVEREXTENSION_WAIT → BUY_ON_PULLBACK，等降溫再進（回測：報酬約持平，±1%）。
OVEREXTENSION_FREE = 55.0
OVEREXTENSION_PENALTY = 0.0
OVEREXTENSION_WAIT = 60.0
PRICE_ONLY_COMPONENTS = ("momentum_rank",)


def regime_weights(profile: Optional[Dict], exclude: tuple = ()) -> Dict[str, float]:
    p = profile or {}
    w = dict(BASE_WEIGHTS)
    w["momentum_rank"] *= p.get("momentum_w", 1.0)
    w["valuation"] *= p.get("valuation_w", 1.0)
    for k in exclude:
        w.pop(k, None)
    return w


def opportunity_score(components: Dict[str, Optional[float]], overextension: Optional[float],
                      weights: Dict[str, float]) -> Dict:
    used = {k: v for k, v in components.items() if k in weights and v is not None and np.isfinite(v)}
    if not used:
        return {"score": None, "coverage": 0.0, "penalty": 0.0, "raw": None}
    tot = sum(weights[k] for k in used)
    raw = sum(used[k] * weights[k] for k in used) / tot
    pen = max(0.0, (overextension or 0) - OVEREXTENSION_FREE) * OVEREXTENSION_PENALTY if overextension is not None else 0.0
    return {"score": round(max(0.0, min(100.0, raw - pen)), 1), "raw": round(raw, 1), "penalty": round(pen, 1),
            "coverage": round(tot / sum(weights.values()), 2), "components_used": sorted(used)}


def opportunity_panel(comp_panels: Dict[str, pd.DataFrame], overext: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """回測用的向量化版本（同一公式）。"""
    num, den = 0, 0
    for k, w in weights.items():
        if k not in comp_panels:
            continue
        p = comp_panels[k]
        num = num + p.fillna(0) * w
        den = den + p.notna().astype(float) * w
    raw = num / den.replace(0, np.nan)
    pen = ((overext - OVEREXTENSION_FREE).clip(lower=0) * OVEREXTENSION_PENALTY).fillna(0)
    return (raw - pen).clip(0, 100)


# ---------------------------------------------------------------------------
# 決策常數（集中在這裡，不散落各處）
# ---------------------------------------------------------------------------
OX_EXTENDED, OX_OVERHEATED, OX_EXTREME = 50.0, 70.0, 85.0      # 與 features.OVEREXTENSION_TIERS 一致
OX_NORMAL_MAX = 25.0
ACCEPTABLE_ABOVE_BUY1_ATR = 0.75     # 現價高於 Buy1 上緣不超過 0.75 ATR、且過熱 < 25 → 仍可立即進場（會明確標示）
GOOD_OPPORTUNITY = 60.0              # 「好公司/好股票」：Opportunity ≥ 60，或品質 ≥ 65 且成長 ≥ 60
BUY_NOW_OPPORTUNITY = 65.0
PULLBACK_OPPORTUNITY = 55.0
MAX_RISK_FOR_BUY_NOW = 90.0
DAMAGE_BLOCK = 60.0                  # 基本面受損 ≥ 60 → 不買、持有者賣
DRAWDOWN_MIN = -0.20                 # 距 52 週高點跌超過 20% 才做回撤分類
BUYABLE_DRAWDOWNS = ("FUNDAMENTAL_DISCOUNT", "TEMPORARY_SHOCK")
THEME_CAP = 0.35                     # 單一主題曝險上限（Σ 權重 × 曝險）
MAX_SINGLE = 0.12
MAX_HIGH_BETA_BUCKET = 0.50


def is_good_company(opp: Optional[float], quality: Optional[float], growth: Optional[float]) -> bool:
    return bool((opp is not None and opp >= GOOD_OPPORTUNITY) or
                (quality is not None and growth is not None and quality >= 65 and growth >= 60))


# ---------------------------------------------------------------------------
# B. Entry Score / Entry Condition
# ---------------------------------------------------------------------------
def entry_condition(price: Optional[float], zones: Dict, overextension: Optional[float],
                    low_entry_price: Optional[float] = None) -> Dict:
    """現價是否可「立即」進場（BUY_NOW 的必要條件）。三種情況才成立，且一定附理由：
      1. 現價 ≤ Buy1 上緣；
      2. 高於 Buy1 不超過 ACCEPTABLE_ABOVE_BUY1_ATR 個 ATR、且過熱 < 25（明確標示「高於偏好區但仍可接受」）；
      3. 現價 ≤ 52 週高點 -30% 的已驗證低檔線（長線低檔布局的進場條件）。
    現價高於 avoid_above 一律不成立。"""
    z1 = (zones or {}).get("zone_1") or {}
    atr = (zones or {}).get("atr")
    if price is None or not z1 or not atr:
        return {"ok": False, "reason": "資料不足：無 Buy1 或 ATR", "atr_above_buy1": None, "basis": None}
    above = max(0.0, (price - z1["high"]) / atr)
    avoid = (zones or {}).get("avoid_above")
    if avoid and price > avoid:
        return {"ok": False, "reason": f"現價 {price} 高於 avoid_above {avoid}", "atr_above_buy1": round(above, 2), "basis": None}
    if price <= z1["high"]:
        return {"ok": True, "reason": f"現價 {price} 在 Buy1（≤ {z1['high']}）內", "atr_above_buy1": 0.0, "basis": "IN_BUY1"}
    if low_entry_price is not None and price <= low_entry_price:
        return {"ok": True, "reason": f"現價 {price} 低於已驗證低檔線 {low_entry_price}（52 週高點 -30%）",
                "atr_above_buy1": round(above, 2), "basis": "BELOW_LOW_ENTRY_LINE"}
    if above <= ACCEPTABLE_ABOVE_BUY1_ATR and (overextension or 0) < OX_NORMAL_MAX:
        return {"ok": True, "reason": (f"current price is still acceptable despite being above preferred entry zone："
                                       f"高於 Buy1 {above:.2f} ATR（≤ {ACCEPTABLE_ABOVE_BUY1_ATR}）且未過熱"),
                "atr_above_buy1": round(above, 2), "basis": "ACCEPTABLE_ABOVE_BUY1"}
    return {"ok": False, "reason": f"現價高於 Buy1 上緣 {z1['high']} 約 {above:.1f} ATR：等回到 Buy1",
            "atr_above_buy1": round(above, 2), "basis": None}


def entry_score(cond: Dict, low_entry: Optional[float], valuation: Optional[float], dist_ma50: Optional[float]) -> Optional[float]:
    """0~100（高 = 現在的價格位置好）：距 Buy1（45%）、已驗證低檔深度（30%）、現價估值（15%）、距 50 日線（10%）。"""
    a = cond.get("atr_above_buy1")
    parts = {
        "location": (None if a is None else max(0.0, 100.0 - a / 3.0 * 100.0), 0.45),
        "low_entry": (low_entry, 0.30),
        "valuation": (valuation, 0.15),
        "ma50": (None if dist_ma50 is None else float(np.clip((0.15 - dist_ma50) / 0.20 * 100, 0, 100)), 0.10),
    }
    used = {k: v for k, v in parts.items() if v[0] is not None}
    if not used:
        return None
    return round(sum(v * w for v, w in used.values()) / sum(w for _, w in used.values()), 1)


# ---------------------------------------------------------------------------
# D. Risk：價格風險 + 基本面不確定性
# ---------------------------------------------------------------------------
def fundamental_uncertainty(m: Optional[Dict]) -> Optional[Dict]:
    """m：operating_margin、fcf_margin、revenue、net_debt_to_ebitda、eps_revision_90d、days_to_earnings。"""
    if not m or all(m.get(k) is None for k in ("operating_margin", "fcf_margin", "revenue")):
        return None
    score, why = 0.0, []
    om, fcf, rev = m.get("operating_margin"), m.get("fcf_margin"), m.get("revenue")
    if om is not None and om < 0:
        score += 30; why.append(f"營業利益率 {om:.0%}（虧損）")
    if fcf is not None and fcf < -0.05:
        score += 25; why.append(f"FCF 率 {fcf:.0%}（燒錢）")
    if rev is None or rev < 100e6:
        score += 25; why.append("營收極小或尚無營收（商業化未證實）")
    nd = m.get("net_debt_to_ebitda")
    if nd is not None and nd > 4:
        score += 10; why.append(f"淨負債/EBITDA {nd:.1f}")
    rv = m.get("eps_revision_90d")
    if rv is not None and rv < -0.05:
        score += 10; why.append(f"EPS 預估 90 天下修 {rv:.0%}")
    dte = m.get("days_to_earnings")
    if dte is not None and 0 <= dte <= 10:
        score += 10; why.append(f"{dte} 天後公布財報（事件風險）")
    return {"score": round(min(100.0, score)), "evidence": why, "speculative": bool((om is not None and om < 0) or
                                                                                  (rev is None or rev < 100e6))}


def combined_risk(price_risk: Optional[float], fu: Optional[Dict]) -> Optional[float]:
    if price_risk is None:
        return None if fu is None else float(fu["score"])
    if fu is None:
        return round(price_risk, 1)
    return round(0.6 * price_risk + 0.4 * fu["score"], 1)


# ---------------------------------------------------------------------------
# 回撤分類：跌很多 ≠ 便宜
# ---------------------------------------------------------------------------
def classify_drawdown(dd_52w: Optional[float], quality: Optional[float], growth: Optional[float],
                      earnings_accel: Optional[float], valuation: Optional[float], damage: Optional[float],
                      fu: Optional[Dict], eps_revision_90d: Optional[float] = None,
                      shock_causes: Optional[List[str]] = None) -> Dict:
    """FUNDAMENTAL_DISCOUNT / TEMPORARY_SHOCK / VALUATION_RESET / SPECULATIVE_DE_RATING / FUNDAMENTAL_DAMAGE / UNKNOWN / NONE。
    只有 FUNDAMENTAL_DISCOUNT、TEMPORARY_SHOCK（且受損低）允許大跌 → BUY_NOW。
    原則：價格證據（公司特有跌幅、跌後破底、關鍵字新聞）只能「提示」受損，要有基本面確認（預估下修、財報/成長轉弱）
    才判 FUNDAMENTAL_DAMAGE；價格與新聞證據極強（受損 ≥ 75）時例外。"""
    if dd_52w is None or dd_52w > DRAWDOWN_MIN:
        return {"type": "NONE", "why": "距 52 週高點跌幅 < 20%，不做回撤分類"}
    rev = eps_revision_90d
    fund_weak = ((rev is not None and rev <= -0.03) or (earnings_accel is not None and earnings_accel < 40)
                 or (growth is not None and growth < 40))
    fund_intact = ((earnings_accel is not None and earnings_accel >= 50) and (rev is None or rev >= 0)
                   and (quality is None or quality >= 50))
    if (damage is not None and damage >= 75) or (damage is not None and damage >= 50 and fund_weak) \
            or (rev is not None and rev <= -0.10) \
            or (growth is not None and earnings_accel is not None and growth < 25 and earnings_accel < 30 and rev is not None and rev <= -0.03):
        return {"type": "FUNDAMENTAL_DAMAGE",
                "why": "基本面受損：大跌伴隨基本面確認（預估下修／財報、成長轉弱）或極強的負面證據"}
    have_fund = any(v is not None for v in (quality, growth, earnings_accel)) or fu is not None
    if not have_fund:
        return {"type": "UNKNOWN", "why": "無基本面資料，無法判斷跌的是價格還是價值"}
    if fu and fu.get("speculative"):
        return {"type": "SPECULATIVE_DE_RATING",
                "why": "虧損或尚無營收：股價反映的是未來預期，下跌是預期修正，不代表便宜（" + "；".join(fu.get("evidence", [])[:2]) + "）"}
    causes = set(shock_causes or [])
    if causes and (damage or 0) < 40 and (causes <= {"macro", "industry", "technical"} or fund_intact):
        return {"type": "TEMPORARY_SHOCK",
                "why": "近期大跌" + ("主因為大盤/產業/技術面" if causes <= {"macro", "industry", "technical"} else "為公司特有，但財報動能與預估未轉差")
                       + "：基本面未受損"}
    if valuation is not None and valuation < 35:
        return {"type": "VALUATION_RESET", "why": f"跌深後估值分數仍僅 {valuation:.0f}：是昂貴估值的修正，尚未便宜"}
    if (quality or 0) >= 50 and (earnings_accel is None or earnings_accel >= 35) and not fund_weak:
        return {"type": "FUNDAMENTAL_DISCOUNT", "why": "品質與財報動能仍在、估值不貴：價格下跌造成的折價"}
    return {"type": "UNKNOWN", "why": "基本面證據不足以判定為折價或受損"}


# ---------------------------------------------------------------------------
# 第二輪（2026-10-08）：LOW PRICE ≠ BUY
#   A. Price Location（只回答「是不是跌深」）
#   B. Investment Quality Tier（由排名決定「低檔規則可以多強」；不改 Rank 本身）
#   C. Low-Price Recommendation（即使跌深，現在值不值得投入新資金）
# ---------------------------------------------------------------------------
PRICE_LOCATIONS = ("DEEPLY_DISCOUNTED", "DISCOUNTED", "PULLBACK", "NEAR_HIGH")
# 以 102 檔美股池的使用者分界（30 / 60 / 85 名）換算成比例，套用到任何大小的股票池（台股 55 檔也適用）
TIER_CUTS = (30 / 102, 60 / 102, 85 / 102)
LOW_PRICE_STATES = ("LOW_PRICE", "LOW_PRICE_OPPORTUNITY", "LOW_PRICE_SPECULATIVE")


def price_location(dd_52w: Optional[float]) -> Optional[str]:
    if dd_52w is None:
        return None
    if dd_52w <= LOW_ENTRY_DD:
        return "DEEPLY_DISCOUNTED"
    if dd_52w <= -0.15:
        return "DISCOUNTED"
    if dd_52w <= -0.05:
        return "PULLBACK"
    return "NEAR_HIGH"


def quality_tier(rank: Optional[int] = None, universe_size: Optional[int] = None,
                 percentile: Optional[float] = None) -> Optional[str]:
    """A（前 30/102）、B（31–60）、C（61–85）、D（> 85）。可用名次/池大小，或動能百分位（100 = 最強）。"""
    if rank is not None and universe_size:
        frac = rank / universe_size
    elif percentile is not None:
        frac = 1 - percentile / 100.0
    else:
        return None
    if frac <= TIER_CUTS[0] + 1e-9:
        return "A"
    if frac <= TIER_CUTS[1] + 1e-9:
        return "B"
    if frac <= TIER_CUTS[2] + 1e-9:
        return "C"
    return "D"


def thesis_confirmations(earnings_accel: Optional[float] = None, eps_revision_90d: Optional[float] = None,
                         catalyst: Optional[float] = None, ret_20: Optional[float] = None,
                         dist_ma50: Optional[float] = None, growth: Optional[float] = None) -> List[str]:
    """與「跌多深」無關的獨立證據（Tier D 要買必須至少一項）。"""
    out = []
    if earnings_accel is not None and earnings_accel >= 65 and (eps_revision_90d is None or eps_revision_90d >= 0):
        out.append("EARNINGS_ACCELERATION")
    if eps_revision_90d is not None and eps_revision_90d >= 0.05 and (growth is None or growth >= 50):
        out.append("FUNDAMENTAL_RECOVERY")
    if catalyst is not None and catalyst >= 65:
        out.append("CATALYST")
    if ret_20 is not None and dist_ma50 is not None and ret_20 >= 0.05 and dist_ma50 > 0:
        out.append("POSITIVE_REVERSAL")
    return out


def low_price_recommendation(location: Optional[str], tier: Optional[str], drawdown_type: Optional[str],
                             confirmations: Optional[List[str]] = None, long_term_winner: bool = True) -> Optional[Dict]:
    """跌深（DEEPLY_DISCOUNTED）的長線贏家：決定低檔狀態與新資金建議。不是跌深就回 None。
    優先序：基本面受損 > 投機下修 > 排名 Tier > 價格位置。跌深本身永遠不是買進理由。
    recommendation：BUY / BUY_STAGED / BUY_ON_PULLBACK / BUY_ON_CONFIRMATION / WATCH / SPECULATIVE_WATCH / NO_BUY"""
    if location != "DEEPLY_DISCOUNTED" or not long_term_winner:
        return None
    dt = drawdown_type or "UNKNOWN"
    conf = list(confirmations or [])
    t = tier or "B"
    buyable = dt in BUYABLE_DRAWDOWNS
    spec_txt = "深度回撤，但目前排名偏低，屬高風險反轉候選；跌深本身不足以構成買進理由。"

    def r(state, rec, label, why):
        return {"state": state, "recommendation": rec, "label": label, "why": why, "tier": t, "drawdown_type": dt,
                "confirmations": conf}

    if dt == "FUNDAMENTAL_DAMAGE":
        return r("LOW_PRICE", "NO_BUY", "跌深但基本面受損：不買", "價格跌深，但回撤分類為基本面受損：跌深不是便宜。")
    if dt == "SPECULATIVE_DE_RATING":
        if t in ("A", "B") and conf:
            return r("LOW_PRICE_SPECULATIVE", "BUY_ON_CONFIRMATION", "高風險反轉候選：小量、等確認",
                     f"投機股預期修正（虧損/營收極小），但有獨立證據（{'、'.join(conf)}）：只能小量、分批。")
        return r("LOW_PRICE_SPECULATIVE", "SPECULATIVE_WATCH", "高風險反轉觀察（不買）",
                 "投機股預期修正（虧損/營收極小）：跌深是預期修正，不代表便宜；只列觀察。" + (spec_txt if t in ("C", "D") else ""))
    if t == "D":
        if conf:
            return r("LOW_PRICE_SPECULATIVE", "BUY_ON_CONFIRMATION", "反轉候選：等確認後小量",
                     f"排名偏低（Tier D），但有獨立證據（{'、'.join(conf)}）：屬反轉候選，確認後小量分批，不是因為跌深而買。")
        return r("LOW_PRICE_SPECULATIVE", "SPECULATIVE_WATCH", "高風險反轉觀察（不買）", spec_txt)
    if t == "C":
        return r("LOW_PRICE", "WATCH", "低檔觀察（不直接買）",
                 "價格跌深，但排名偏後（Tier C）：只列低檔觀察，等排名或基本面改善再考慮。")
    if t == "B":
        if buyable:
            return r("LOW_PRICE_OPPORTUNITY", "BUY_STAGED", "低檔分批（中等排名的跌深機會）",
                     f"低檔＋排名中等（Tier B）＋基本面未明顯破壞（{dt}）：可分批、降低部位，不是因為跌很多而買。")
        return r("LOW_PRICE", "WATCH", "低檔觀察（不直接買）",
                 f"價格跌深、排名中等（Tier B），但回撤分類為 {dt}：基本面證據不足，只列觀察。")
    # Tier A
    if buyable:
        return r("LOW_PRICE_OPPORTUNITY", "BUY", "低檔布局可買（高排名回撤）",
                 f"高排名（Tier A）的長線贏家回撤，回撤分類 {dt}：低檔布局可分批買進。")
    return r("LOW_PRICE_OPPORTUNITY", "BUY_ON_PULLBACK", "高排名回撤：等拉回分批",
             f"高排名（Tier A）回撤，但回撤分類為 {dt}（估值仍不便宜或證據不足）：等拉回到 Buy2 分批。")


# ---------------------------------------------------------------------------
# E. Final Action
# ---------------------------------------------------------------------------
def decide_action(opp: Optional[float], overext: Optional[float], quality: Optional[float], risk: Optional[float],
                  valuation: Optional[float], relative_strength: Optional[float], held: bool,
                  damage: Optional[float] = None, oversold: bool = False, entry_ok: bool = False,
                  low_entry: bool = False, long_term_broken: bool = False, growth: Optional[float] = None,
                  drawdown_type: str = "UNKNOWN", tier: Optional[str] = None,
                  confirmations: Optional[List[str]] = None) -> Dict:
    """回傳 status、state（例如 GOOD_BUT_OVEREXTENDED）、low_price（低檔狀態）、flags、why、pullback_reason。
    優先序：基本面受損 > 極度過熱 > 風險 > 進場條件 > 排名 Tier > 價格位置。
    BUY_NOW 的必要條件：entry_ok、過熱 < 50、非基本面受損、風險 < 90；低檔區另需 Tier A/B 且回撤類型屬可買類。"""
    flags: List[str] = []
    if opp is None:
        return {"status": "WAIT", "state": None, "flags": ["INSUFFICIENT_DATA"], "why": "資料不足，無法評分", "pullback_reason": None}
    ox = overext or 0.0
    good = is_good_company(opp, quality, growth)
    damaged = drawdown_type == "FUNDAMENTAL_DAMAGE" or (damage is not None and damage >= DAMAGE_BLOCK)
    state = "GOOD_BUT_OVEREXTENDED" if (good and ox >= OX_EXTENDED) else None
    if oversold:
        flags.append("POTENTIAL_OVERSOLD")
    if low_entry:
        flags.append("LOW_ENTRY_ZONE")
    if state:
        flags += ["BUY_QUALITY", "WAIT_ENTRY"]
    if damaged:
        flags.append("THESIS_AT_RISK")

    lp = low_price_recommendation("DEEPLY_DISCOUNTED", tier, drawdown_type, confirmations) if low_entry else None

    def out(status, why, pb=None):
        return {"status": status, "state": state, "flags": flags, "why": why, "pullback_reason": pb, "low_price": lp}

    tier = "極度過熱" if ox >= OX_EXTREME else ("過熱" if ox >= OX_OVERHEATED else "偏熱")
    if held:
        if damaged:
            return out("SELL", "基本面受損（回撤分類 FUNDAMENTAL_DAMAGE 或受損分數 ≥ 60）")
        if ox >= OX_EXTREME and (valuation is None or valuation < 50):
            return out("PARTIAL_PROFIT", f"{tier}（{ox:.0f}）且估值不便宜：先分批獲利了結")
        if ox >= OX_OVERHEATED and valuation is not None and valuation < 35:
            return out("PARTIAL_PROFIT", f"{tier}（{ox:.0f}）且估值偏貴（{valuation:.0f}）：先分批獲利了結")
        if low_entry and lp and lp["recommendation"] not in ("NO_BUY", "SPECULATIVE_WATCH"):
            return out("HOLD_CORE", f"長線贏家落入低檔區（{lp['label']}）：續抱（是否加碼看新資金建議）")
        if low_entry and lp:
            return out("HOLD", f"價格跌深但{lp['label']}：續抱、不加碼")
        if quality is not None and quality >= 70 and opp >= 45:
            return out("HOLD_CORE", "高品質核心部位，機會分數仍在中上" + ("；短線過熱，不加碼" if ox >= OX_EXTENDED else ""))
        if opp < 30 and (relative_strength or 50) < 30 and long_term_broken:
            return out("SELL", "機會分數與相對強度同時轉弱，且 3 年長線趨勢已破壞")
        if state == "GOOD_BUT_OVEREXTENDED":
            return out("HOLD", f"好公司＋強動能，但短線{tier}（{ox:.0f}）：持有者續抱，不加碼、新資金不追")
        if opp >= 30 or not long_term_broken:
            return out("HOLD", "續抱觀察" if opp >= 30 else "短線動能弱，但 3 年長線趨勢未破壞：續抱、不加碼")
        return out("SELL", "機會分數偏低且 3 年長線趨勢已破壞")

    # ---- 新資金 ----
    if damaged:
        return out("WAIT", "基本面受損：不承接（大跌不等於便宜）")
    if ox >= OX_EXTREME:
        return out("WAIT", (f"基本面與成長性仍然優秀，但{tier}（{ox:.0f}）：目前不適合追價，等降溫後回到進場區" if good
                            else f"{tier}（{ox:.0f}）且機會分數普通（{opp:.0f}）：不追"))
    if ox >= OX_EXTENDED:
        if good:
            return out("BUY_ON_PULLBACK", f"基本面與成長性仍然優秀，但短期漲幅與技術位置{tier}（{ox:.0f}）：目前不適合追價；"
                                          "等降溫（過熱 < 50）或回撤至 Buy2", "COOLING_REQUIRED")
        return out("WAIT", f"{tier}（{ox:.0f}）且機會分數普通（{opp:.0f}）：不追")
    if low_entry and lp:
        rec = lp["recommendation"]
        risk_ok = risk is None or risk < MAX_RISK_FOR_BUY_NOW
        if rec in ("BUY", "BUY_STAGED") and (damage or 0) < 40:
            if rec == "BUY_STAGED":
                flags.append("STAGED_ENTRY")
            if entry_ok and risk_ok:
                return out("BUY_NOW", lp["why"] + ("（分批、降低部位）" if rec == "BUY_STAGED" else "") + f"持有 12 個月。")
            return out("BUY_ON_PULLBACK", lp["why"] + ("風險過高，只在更深的進場區小量" if entry_ok else "現價不在進場區：等回到 Buy1"),
                       "CONFIRMATION_REQUIRED" if entry_ok else "ABOVE_ENTRY")
        if rec in ("BUY_ON_PULLBACK", "BUY_ON_CONFIRMATION"):
            if rec == "BUY_ON_CONFIRMATION":
                flags.append("REVERSAL_CANDIDATE")
            return out("BUY_ON_PULLBACK", lp["why"], "CONFIRMATION_REQUIRED")
        return out("WAIT", lp["why"])
    if opp >= BUY_NOW_OPPORTUNITY:
        if entry_ok and (risk is None or risk < MAX_RISK_FOR_BUY_NOW):
            return out("BUY_NOW", "機會分數高、現價在可進場區且未過熱")
        if entry_ok:
            return out("BUY_ON_PULLBACK", f"機會分數高但風險 {risk:.0f} 過高：只在更深的進場區小量", "CONFIRMATION_REQUIRED")
        return out("BUY_ON_PULLBACK", "機會分數高，但現價高於進場區：等回到 Buy1", "ABOVE_ENTRY")
    if opp >= PULLBACK_OPPORTUNITY:
        return out("BUY_ON_PULLBACK", "機會分數中上：只在 Buy1 以下分批，不追價", "CONFIRMATION_REQUIRED" if entry_ok else "ABOVE_ENTRY")
    if oversold and opp >= 50 and (damage or 0) < 40:
        return out("BUY_ON_PULLBACK", "大跌但基本面受損低，分批於進場區承接", "CONFIRMATION_REQUIRED")
    if ox >= OX_NORMAL_MAX and opp >= 45:
        return out("WAIT", f"機會分數中等（{opp:.0f}）且短線偏熱（{ox:.0f}）：觀望")
    return out("WAIT", f"機會分數不足（{opp:.0f}），觀望")


def decide_status(opp: Optional[float], overext: Optional[float], quality: Optional[float], risk: Optional[float],
                  valuation: Optional[float], relative_strength: Optional[float], held: bool,
                  damage: Optional[float] = None, oversold: bool = False, above_avoid: bool = False,
                  low_entry: bool = False, long_term_broken: bool = False, entry_ok: bool = False,
                  drawdown_type: str = "UNKNOWN", growth: Optional[float] = None, tier: Optional[str] = None,
                  confirmations: Optional[List[str]] = None) -> Dict:
    """相容舊介面：轉呼叫 decide_action。entry_ok 預設 False——沒有進場條件就不會給 BUY_NOW。"""
    return decide_action(opp, overext, quality, risk, valuation, relative_strength, held, damage, oversold,
                         entry_ok and not above_avoid, low_entry, long_term_broken, growth, drawdown_type, tier, confirmations)


# ---------------------------------------------------------------------------
# 進場區（動態）
# ---------------------------------------------------------------------------
def historical_pullbacks(close: pd.Series, window: int = 60, years: int = 2) -> Dict[str, Optional[float]]:
    """過去 2 年在上升趨勢（站上 200 日線）中，自 60 日高點回檔的深度分布。"""
    c = close.dropna().tail(252 * years + 200)
    if len(c) < 260:
        return {"median": None, "p75": None}
    dd = c / c.rolling(window).max() - 1
    up = c > c.rolling(200).mean()
    troughs = []
    in_dd, low = False, 0.0
    for d, x in dd.items():
        if x < -0.03 and up.get(d, False):
            in_dd, low = True, min(low, x)
        elif in_dd and x > -0.01:
            troughs.append(low)
            in_dd, low = False, 0.0
    if len(troughs) < 3:
        return {"median": None, "p75": None}
    t = np.array(troughs)
    return {"median": float(np.median(t)), "p75": float(np.percentile(t, 25))}   # 25 百分位 = 較深的回檔


def entry_zones(ohlcv: pd.DataFrame, valuation_score: Optional[float] = None,
                days_to_earnings: Optional[int] = None) -> Dict:
    c, h, l = ohlcv["Close"].dropna(), ohlcv["High"].dropna(), ohlcv["Low"].dropna()
    if len(c) < 60:
        return {"zone_1": None, "zone_2": None, "zone_3": None, "avoid_above": None, "notes": ["資料不足"]}
    px = float(c.iloc[-1])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    ma20, ma50 = float(c.tail(20).mean()), float(c.tail(50).mean())
    ma200 = float(c.tail(200).mean()) if len(c) >= 200 else None
    low20, low60 = float(l.tail(20).min()), float(l.tail(60).min())
    hi60 = float(h.tail(60).max())
    lo252, hi252 = float(l.tail(252).min()), float(h.tail(252).max())
    pb = historical_pullbacks(c)
    cands = {"MA20": ma20, "MA50": ma50, "20 日低點": low20, "60 日低點": low60,
             "1 ATR": px - atr, "2 ATR": px - 2 * atr, "3 ATR": px - 3 * atr}
    if ma200:
        cands["MA200"] = ma200
    if pb["median"] is not None:
        cands["歷史中位回檔"] = hi60 * (1 + pb["median"])
        cands["歷史深回檔"] = hi60 * (1 + pb["p75"])
    shift = 0.5 * atr if (valuation_score is not None and valuation_score < 30) else 0.0
    below = sorted([(v - shift, k) for k, v in cands.items() if v < px * 0.995 and v > lo252 * 0.98], reverse=True)

    def pick(lo_bound: float, default: float, default_name: str):
        for v, k in below:
            if v >= lo_bound:
                return v, k
        return default, default_name

    z1, n1 = pick(px - 1.5 * atr, px - atr - shift, "1 ATR")
    rest = [(v, k) for v, k in below if v < z1 - 0.5 * atr]
    z2, n2 = next(((v, k) for v, k in rest if v >= px - 3.5 * atr), (z1 - atr, f"{n1}−1 ATR"))
    deeper = [(v, k) for v, k in below if v < z2 - 0.5 * atr]
    z3, n3 = deeper[0] if deeper else (max(z2 - 1.5 * atr, lo252), "52 週區間下緣")
    if z3 > z2 - 0.5 * atr:                       # 排序保證：Buy1 > Buy2 > Deep
        z3, n3 = z2 - 1.5 * atr, f"{n2}−1.5 ATR"
    band = 0.25 * atr

    def zone(v, name):
        return {"low": round(v - band, 2), "high": round(v + band, 2), "basis": name,
                "pct_from_price": round((v / px - 1) * 100, 1)}

    avoid = min(ma20 + 2.5 * atr, max(px, hi252) * 1.02)
    notes = []
    if shift:
        notes.append("估值偏貴：各區下移 0.5 ATR")
    if days_to_earnings is not None and 0 <= days_to_earnings <= 10:
        notes.append(f"{days_to_earnings} 天後公布財報：Zone 1 只建小部位，財報後再決定")
    if px > avoid:
        notes.append("現價已高於 avoid_above，不追價")
    return {"zone_1": zone(z1, n1), "zone_2": zone(z2, n2), "zone_3": zone(z3, n3),
            "avoid_above": round(avoid, 2), "atr": round(atr, 2), "price": round(px, 2),
            "range_52w": [round(lo252, 2), round(hi252, 2)], "notes": notes}


# ---------------------------------------------------------------------------
# 預期報酬、輪動、部位
# ---------------------------------------------------------------------------
# 機會分數 → 預期 60 日超額報酬。預設為保守線性假設；P2 校準後由回測結果取代（calibration 檔）。
DEFAULT_CALIBRATION = {"slope_per_point": 0.0015, "center": 50.0, "source": "default linear prior (未校準)"}
ROUND_TRIP_COST = 0.004


def expected_excess_return(opp: Optional[float], calibration: Optional[Dict] = None) -> Optional[float]:
    if opp is None:
        return None
    cal = calibration or DEFAULT_CALIBRATION
    if "deciles" in cal:   # 由回測得到的分位數表
        edges, vals = cal["deciles"]["edges"], cal["deciles"]["mean_excess_60d"]
        i = int(np.searchsorted(edges, opp, side="right")) - 1
        return float(vals[max(0, min(i, len(vals) - 1))])
    return (opp - cal["center"]) * cal["slope_per_point"]


def rotation_suggestions(rows: List[Dict], calibration: Optional[Dict] = None, max_pairs: int = 5) -> List[Dict]:
    """SELL A → BUY B 的成對建議。rotation_score = 風險調整後的預期超額報酬差，再依估值/動能/催化調整。"""
    held = [r for r in rows if r.get("held")]
    cands = [r for r in rows if not r.get("held") and r["status"] in ("BUY_NOW", "BUY_ON_PULLBACK")]
    out = []
    for a in held:
        ea = expected_excess_return(a.get("opportunity"), calibration)
        if ea is None:
            continue
        for b in cands:
            eb = expected_excess_return(b.get("opportunity"), calibration)
            if eb is None:
                continue
            va, vb = max(a.get("vol_63") or 0.4, 0.1), max(b.get("vol_63") or 0.4, 0.1)
            diff = eb / vb * 0.40 - ea / va * 0.40           # 換算成「40% 年化波動」的同風險基準
            adj = 0.0
            adj += 0.01 if (b["scores"].get("valuation") or 50) > (a["scores"].get("valuation") or 50) + 15 else 0.0
            adj += 0.01 if (b["scores"].get("catalyst") or 50) > (a["scores"].get("catalyst") or 50) + 15 else 0.0
            adj -= 0.01 if (b["scores"].get("overextension") or 0) >= 60 else 0.0
            score = diff + adj - ROUND_TRIP_COST
            if score <= 0.02:
                continue
            kind = "FULL_ROTATION" if (a["status"] == "SELL" or score > 0.08) else "PARTIAL_ROTATION"
            out.append({"sell": a["ticker"], "buy": b["ticker"], "rotation_score": round(score, 4), "type": kind,
                        "fraction_of_A": 1.0 if kind == "FULL_ROTATION" else 0.33,
                        "why": f"{b['ticker']} 風險調整後預期 60 日超額 {eb:+.1%} vs {a['ticker']} {ea:+.1%}"})
    out.sort(key=lambda x: x["rotation_score"], reverse=True)
    return out[:max_pairs]


def theme_exposure(weights: Dict[str, float], exposures: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Σ 部位權重 × 主題曝險權重。"""
    out: Dict[str, float] = {}
    for k, w in weights.items():
        for th, e in (exposures.get(k) or {"Other": 1.0}).items():
            out[th] = out.get(th, 0.0) + w * e
    return out


def position_sizes(rows: List[Dict], profile: Dict, calibration: Optional[Dict] = None,
                   max_single: float = MAX_SINGLE, theme_cap: float = THEME_CAP,
                   max_high_beta_bucket: float = MAX_HIGH_BETA_BUCKET,
                   exposures: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """組合層級：目標權重 ∝ 預期超額報酬 / 預期風險 × 資料覆蓋；再套單檔、高 beta、主題曝險上限。
    exposures：{ticker: {主題: 權重}}（themes.theme_exposures）；缺則用 row["theme"] 當單一主題。
    只回傳權重，不修改任何 row 的分數（Score 是股票本身，Sizing 是組合脈絡）。"""
    raw = {}
    for r in rows:
        if r["status"] not in ("BUY_NOW", "BUY_ON_PULLBACK", "HOLD", "HOLD_CORE"):
            continue
        e = expected_excess_return(r.get("opportunity"), calibration)
        if e is None or e <= 0:
            continue
        vol = max(r.get("vol_63") or 0.4, 0.15)
        downside = max(r.get("downside_risk") or vol, 0.10)
        raw[r["ticker"]] = e / (0.5 * vol + 0.5 * downside) * (r.get("coverage") or 0.5)
    if not raw:
        return {}
    tot = sum(raw.values())
    w = {k: v / tot * profile.get("exposure", 1.0) for k, v in raw.items()}
    hb_cap = profile.get("max_high_beta_weight", max_single)
    beta = {r["ticker"]: r.get("beta") or 1.0 for r in rows}
    exp = {r["ticker"]: (exposures or {}).get(r["ticker"]) or {(r.get("theme") or "Other"): 1.0} for r in rows}
    for _ in range(20):   # 反覆套上限（主題縮減會影響其他主題，迭代到收斂）
        capped = {k: min(v, min(max_single, hb_cap) if beta.get(k, 1) >= 1.5 else max_single) for k, v in w.items()}
        te = theme_exposure(capped, exp)
        for th, tv in te.items():
            if tv > theme_cap + 1e-12:
                f = theme_cap / tv
                for k in capped:
                    ek = exp[k].get(th, 0.0)
                    if ek > 0:
                        capped[k] *= 1 - ek * (1 - f)   # 曝險越高的股票縮越多
        hb = sum(v for k, v in capped.items() if beta.get(k, 1) >= 1.5)
        if hb > max_high_beta_bucket:
            for k in capped:
                if beta.get(k, 1) >= 1.5:
                    capped[k] *= max_high_beta_bucket / hb
        done = all(abs(capped[k] - w[k]) < 1e-9 for k in w)
        w = capped
        if done:
            break
    return {k: round(v, 4) for k, v in sorted(w.items(), key=lambda x: -x[1]) if v >= 0.005}
