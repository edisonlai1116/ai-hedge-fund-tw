"""Opportunity Score（價格模組）的嚴格 point-in-time 回測與驗證工具。

時序：T-1 收盤算分數 → T 開盤成交 → 持有到下一次調整的開盤。報酬一律 open-to-open。
只回測價格模組（相對強度、技術、產業動能、風險、過熱扣分）：基本面/新聞/AI 曝險沒有 point-in-time
歷史，放進來就是作弊，所以不放。

提供：
  run_strategy        單一策略回測（含換股紀錄、交易統計）
  metrics             CAGR / Sharpe / Sortino / MDD / 勝率 / Profit Factor / 換手 / 平均持有 / 對 SPY、QQQ alpha
  market_states       多頭 / 空頭 / 盤整（依 SPY 的 200 日線與 6 個月報酬，point-in-time）
  walk_forward        每年用過去 3 年選參數 → 套用下一年（樣本外）
  ablation / sensitivity / ticker_concentration（OVERFIT 檢查）
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ranking import features as fx
from src.ranking import scoring as sc

COST = 0.002          # 每次換一檔（買或賣）的成本
IS_END = "2021-12-31"
OOS_START = "2022-01-01"


# ---------------------------------------------------------------------------
# 分數面板
# ---------------------------------------------------------------------------
def score_panels(F: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    risk = fx.risk_score(F)
    return {
        "momentum_rank": fx.pct_rank(F["sharpe_126"]),
        "relative_strength": fx.relative_strength_score(F),
        "technical": fx.technical_score(F),
        "industry_momentum": fx.industry_momentum_score(F),
        "low_risk": 100 - risk,
        "overextension": fx.overextension_score(F),
        "sharpe_126": F["sharpe_126"],
    }


def opportunity_from(panels: Dict[str, pd.DataFrame], exclude: tuple = (), penalty: float = sc.OVEREXTENSION_PENALTY,
                     free: float = sc.OVEREXTENSION_FREE, weights: Optional[Dict[str, float]] = None,
                     components: tuple = sc.PRICE_ONLY_COMPONENTS) -> pd.DataFrame:
    w = {k: v for k, v in (weights or sc.BASE_WEIGHTS).items() if k in components and k not in exclude}
    num = sum(panels[k].fillna(0) * v for k, v in w.items())
    den = sum(panels[k].notna().astype(float) * v for k, v in w.items()).replace(0, np.nan)
    raw = num / den
    if "overextension" in exclude or penalty == 0:
        return raw
    return (raw - ((panels["overextension"] - free).clip(lower=0) * penalty).fillna(0)).clip(0, 100)


# ---------------------------------------------------------------------------
# 回測
# ---------------------------------------------------------------------------
def run_strategy(score: pd.DataFrame, open_px: pd.DataFrame, top: int = 10, keep: int = 30, reb: int = 21,
                 start: str = "2017-07-01", exposure: Optional[pd.Series] = None,
                 overext: Optional[pd.DataFrame] = None, wait_above: Optional[float] = None) -> Dict:
    """score：T 日收盤後可得的分數（列 = 日期）。T+1 開盤依 score.loc[T] 調整。
    overext/wait_above：新進場若過熱 ≥ wait_above，名額先留現金，之後每天檢查，降溫才買（不追高）。
    使用 wait 規則時每檔固定 1/top 權重（未買進的名額為現金）。"""
    score = score.reindex(open_px.index)
    oo = open_px.shift(-1) / open_px - 1                      # 從今天開盤持有到明天開盤
    idx = open_px.index[open_px.index >= start]
    held: List[str] = []
    pending: List[str] = []
    entry: Dict[str, Tuple[pd.Timestamp, float]] = {}
    trades: List[Dict] = []
    eq, dates, turnover = [1.0], [idx[0]], 0
    for j in range(1, len(idx) - 1):
        d, prev = idx[j], idx[j - 1]
        cost = 0.0
        if (j - 1) % reb == 0:
            s = score.loc[prev].dropna().sort_values(ascending=False)
            pos = {k: i for i, k in enumerate(s.index)}
            stay = [h for h in held if pos.get(h, 10 ** 9) < keep]
            new = [k for k in s.index if k not in stay][: max(0, top - len(stay))]
            if wait_above is not None and overext is not None:
                pending = [k for k in new if overext.at[prev, k] >= wait_above] if prev in overext.index else []
                new = [k for k in new if k not in pending]
            nxt = stay + new
            for k in set(held) - set(nxt):
                t0, p0 = entry.pop(k)
                p1 = open_px.at[d, k]
                if np.isfinite(p1) and p0 > 0:
                    trades.append({"ticker": k, "entry": t0, "exit": d, "return": p1 / p0 - 1 - 2 * COST,
                                   "days": (d - t0).days})
            for k in set(nxt) - set(held):
                entry[k] = (d, open_px.at[d, k])
            ch = len(set(nxt) ^ set(held))
            turnover += ch
            cost = COST * ch / top
            held = nxt
        elif pending:
            ok = [k for k in pending if overext.at[prev, k] < wait_above]
            for k in ok:
                entry[k] = (d, open_px.at[d, k])
            if ok:
                held = held + ok
                pending = [k for k in pending if k not in ok]
                turnover += len(ok)
                cost = COST * len(ok) / top
        if wait_above is not None:
            r = oo.loc[d, held].sum() / top if held else 0.0
        else:
            r = oo.loc[d, held].mean() if held else 0.0
        r = 0.0 if not np.isfinite(r) else float(r)
        x = float(exposure.get(prev, 1.0)) if exposure is not None else 1.0
        eq.append(eq[-1] * (1 + x * r - cost))
        dates.append(d)
    for k, (t0, p0) in entry.items():          # 期末未平倉部位以最後開盤價結算
        p1 = open_px[k].dropna().iloc[-1]
        trades.append({"ticker": k, "entry": t0, "exit": idx[-1], "return": p1 / p0 - 1 - COST, "days": (idx[-1] - t0).days})
    years = (dates[-1] - dates[0]).days / 365.25
    return {"equity": pd.Series(eq, index=dates), "trades": trades, "turnover_per_year": turnover / max(years, 1e-9),
            "final_holdings": held}


def _ann_alpha(r: pd.Series, b: pd.Series) -> Optional[float]:
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 60 or df.iloc[:, 1].var() == 0:
        return None
    beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
    return float((df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * 252)


def metrics(eq: pd.Series, trades: Optional[List[Dict]] = None, bench: Optional[Dict[str, pd.Series]] = None,
            turnover: Optional[float] = None) -> Dict:
    eq = eq.dropna()
    eq = eq / eq.iloc[0]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    r = eq.pct_change().dropna()
    dn = r[r < 0]
    out = {
        "CAGR": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
        "Sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() else None,
        "Sortino": round(float(r.mean() / dn.std() * math.sqrt(252)), 2) if len(dn) > 2 and dn.std() else None,
        "MaxDD": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
    }
    if trades:
        t = [x for x in trades if eq.index[0] <= x["exit"] <= eq.index[-1]]
        rets = np.array([x["return"] for x in t]) if t else np.array([])
        gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
        out.update({
            "WinRate": round(float((rets > 0).mean()) * 100, 1) if len(rets) else None,
            "ProfitFactor": round(float(gains / losses), 2) if losses > 0 else None,
            "AvgHoldDays": round(float(np.mean([x["days"] for x in t])), 0) if t else None,
            "Trades": len(t),
        })
    if turnover is not None:
        out["TurnoverPerYear"] = round(turnover, 1)
    for name, b in (bench or {}).items():
        bb = b.reindex(eq.index).ffill()
        a = _ann_alpha(r, bb.pct_change().dropna())
        out[f"Alpha_vs_{name}"] = None if a is None else round(a * 100, 1)
    return out


def split_metrics(res: Dict, bench_open: Dict[str, pd.Series]) -> Dict[str, Dict]:
    eq, tr = res["equity"], res["trades"]
    out = {}
    for lab, (a, z) in {"全期": (None, None), "In-sample 2017-2021": (None, IS_END), "Out-of-sample 2022-": (OOS_START, None)}.items():
        e = eq[a:z]
        if len(e) > 60:
            out[lab] = metrics(e, tr, bench_open, res["turnover_per_year"] if lab == "全期" else None)
    return out


def market_states(spy: pd.Series) -> pd.Series:
    """point-in-time 市場狀態（只用當天以前的 SPY）：bull / bear / sideways。"""
    ma = spy.rolling(200).mean()
    r6 = spy / spy.shift(126) - 1
    st = pd.Series("sideways", index=spy.index)
    st[(spy > ma) & (r6 > 0.05)] = "bull"
    st[(spy < ma) & (r6 < -0.05)] = "bear"
    return st.where(ma.notna())


def state_metrics(eq: pd.Series, states: pd.Series, bench: pd.Series) -> Dict[str, Dict]:
    r = eq.pct_change().dropna()
    b = bench.reindex(eq.index).ffill().pct_change().reindex(r.index)
    s = states.shift(1).reindex(r.index)      # 用前一天的狀態標記今天
    out = {}
    for name in ("bull", "bear", "sideways"):
        m = s == name
        if m.sum() < 40:
            continue
        rr, bb = r[m], b[m]
        out[name] = {"days": int(m.sum()), "ann_return_pct": round(float(rr.mean() * 252 * 100), 1),
                     "bench_ann_return_pct": round(float(bb.mean() * 252 * 100), 1),
                     "sharpe": round(float(rr.mean() / rr.std() * math.sqrt(252)), 2) if rr.std() else None,
                     "hit_rate_vs_bench_pct": round(float((rr > bb).mean() * 100), 1)}
    return out


def walk_forward(candidates: Dict[str, pd.DataFrame], open_px: pd.DataFrame, top: int, keep: int,
                 first_test_year: int = 2020, train_years: int = 3) -> Dict:
    """每年：用過去 train_years 年的 Sharpe 選最佳候選策略，套用到下一年（真正樣本外），串接成一條權益曲線。"""
    pieces, picks = [], {}
    last_year = open_px.index[-1].year
    for y in range(first_test_year, last_year + 1):
        tr_start, tr_end = f"{y - train_years}-01-01", f"{y - 1}-12-31"
        best, best_sh = None, -9
        for name, s in candidates.items():
            e = run_strategy(s, open_px, top, keep, start=tr_start)["equity"][:tr_end]
            rr = e.pct_change().dropna()
            sh = rr.mean() / rr.std() if len(rr) > 60 and rr.std() else -9
            if sh > best_sh:
                best, best_sh = name, sh
        picks[y] = best
        e = run_strategy(candidates[best], open_px, top, keep, start=f"{y - 1}-10-01")["equity"][f"{y}-01-01":f"{y}-12-31"]
        if len(e) > 5:
            pieces.append(e.pct_change().fillna(0))
    r = pd.concat(pieces)
    return {"equity": (1 + r).cumprod(), "picks": picks}


def ticker_concentration(trades: List[Dict], top_k: int = 1) -> Dict:
    """各股對總報酬的貢獻；若拿掉貢獻最大的 1 檔後總報酬轉負 → OVERFIT（只靠單一股票）。"""
    by: Dict[str, float] = {}
    for t in trades:
        by[t["ticker"]] = by.get(t["ticker"], 0) + math.log1p(max(t["return"], -0.99))
    total = sum(by.values())
    top = sorted(by.items(), key=lambda x: -x[1])[:5]
    without = total - sum(v for _, v in top[:top_k])
    share = (top[0][1] / total) if (top and total > 0) else None
    return {"total_log_return": round(total, 3), "top_contributors": [(k, round(v, 3)) for k, v in top],
            "top1_share": None if share is None else round(share, 2),
            "without_top1_log_return": round(without, 3),
            "verdict": "OVERFIT" if (total > 0 and without <= 0) else ("CONCENTRATED" if (share or 0) > 0.4 else "OK")}


def calibration_table(score: pd.DataFrame, close: pd.DataFrame, bench: pd.Series, start: str = OOS_START,
                      horizon: int = 60, step: int = 5) -> Dict:
    """分數十分位 → 未來 60 日平均超額報酬（樣本外期間），供即時輪動/部位換算預期報酬。"""
    fwd = close.shift(-horizon) / close - 1
    bf = (bench.shift(-horizon) / bench - 1).reindex(close.index)
    ex = fwd.sub(bf, axis=0)
    rows = []
    for d in score.index[(score.index >= start)][::step]:
        s, e = score.loc[d], ex.loc[d]
        m = s.notna() & e.notna()
        rows += list(zip(s[m].values, e[m].values))
    if len(rows) < 200:
        return {}
    df = pd.DataFrame(rows, columns=["s", "e"])
    df["q"] = pd.qcut(df["s"], 10, labels=False, duplicates="drop")
    g = df.groupby("q").agg(lo=("s", "min"), mean=("e", "mean"), n=("e", "size"))
    return {"deciles": {"edges": [round(float(x), 2) for x in g["lo"].tolist()],
                        "mean_excess_60d": [round(float(x), 4) for x in g["mean"].tolist()],
                        "n": [int(x) for x in g["n"].tolist()]},
            "source": f"out-of-sample {start}~ 價格版 Opportunity Score 十分位，未來 {horizon} 日超額報酬"}
