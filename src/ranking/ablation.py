"""Release 驗收用消融測試：python -m src.ranking.ablation --market us|tw

全部 point-in-time：T-1 收盤分數 → T 開盤成交，open-to-open，每次換股成本 0.2%。
  Model A  原本 analyst voting（舊網站 simple_signal 綜合分數：趨勢/動能/突破/回檔/風險 agent 投票 + 規則分數），
           每個調整日只用當日以前的價格重算（lightweight，不含無歷史時點的基本面）
  Model B  量化因子（相對強度、技術、產業動能、低風險）
  Model C  B + 過熱扣分（catalyst 無 point-in-time 歷史 → 無法納入，明確標示）
  Model D  完整模型：70% 長線低檔布局 + 30% 動能排名（過熱等降溫）、跌出緩衝才換股（輪動）
結果寫入 docs/data/ablation_{market}.json。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.ranking import backtest as bt
from src.ranking import features as fx
from src.ranking.themes import industry_benchmark, theme_of
from src.ranking.validate import BENCH, STRATEGY_PARAMS

START = "2017-07-01"
REB = 21


def lowentry_open(close, open_px, start=START, hold=252, slots=10):
    """長線低檔布局（open-to-open）：T-1 收盤符合 → T 開盤買進，持有 hold 日。"""
    dd = close / close.rolling(252, min_periods=200).max() - 1
    ok = (dd <= -0.30) & (close / close.shift(756) - 1 > 0)
    oo = open_px.shift(-1) / open_px - 1
    idx = open_px.index[open_px.index >= start]
    pos, eq, trades, cool, turnover = {}, [1.0], [], {}, 0
    for j in range(1, len(idx) - 1):
        d, prev = idx[j], idx[j - 1]
        cost = 0.0
        for s in list(pos):
            j0, p0 = pos[s]
            if j - j0 >= hold:
                px = open_px.at[d, s]
                if np.isfinite(px):
                    trades.append({"ticker": s, "entry": idx[j0], "exit": d, "return": px / p0 - 1 - 2 * bt.COST, "days": (d - idx[j0]).days})
                del pos[s]; cool[s] = j + 60; cost += bt.COST / slots; turnover += 1
        if len(pos) < slots:
            row = ok.loc[prev]
            c = sorted([k for k in row[row.fillna(False)].index if k not in pos and cool.get(k, 0) <= j], key=lambda k: dd.at[prev, k])
            for k in c[: slots - len(pos)]:
                if np.isfinite(open_px.at[d, k]):
                    pos[k] = (j, open_px.at[d, k]); cost += bt.COST / slots; turnover += 1
        r = sum(oo.at[d, s] for s in pos if np.isfinite(oo.at[d, s])) / slots
        eq.append(eq[-1] * (1 + r - cost))
    yrs = (idx[-1] - idx[0]).days / 365.25
    return {"equity": pd.Series(eq, index=idx[:len(eq)]), "trades": trades, "turnover_per_year": turnover / yrs}


def model_a_scores(pm, open_px, rebal_prev_dates):
    """舊系統 analyst voting 分數（只在調整日計算，每次只餵當日以前的資料）。"""
    from src.simple_signal import build_report
    rows = {}
    for d in rebal_prev_dates:
        vals = {}
        for s, f in pm.items():
            g = f[f.index <= d]
            if len(g) < 260:
                continue
            try:
                vals[s] = float(build_report(s, g.tail(520), fetch_fundamentals=False, lightweight=True).composite_score)
            except Exception:
                continue
        rows[d] = vals
    return pd.DataFrame(rows).T.reindex(open_px.index)


def combine(results, weights):
    eqs = {k: r["equity"] for k, r in results.items()}
    idx = sorted(set().union(*[set(e.index) for e in eqs.values()]))
    rr = sum(w * eqs[k].reindex(idx).ffill().pct_change().fillna(0) for k, w in weights.items())
    trades = sum((results[k]["trades"] for k in weights), [])
    to = sum(w * results[k]["turnover_per_year"] for k, w in weights.items())
    return {"equity": (1 + rr).cumprod(), "trades": trades, "turnover_per_year": to}


def run(market: str) -> dict:
    from src.strategy.momentum import download_closes, universe
    pm = download_closes(universe(market) + BENCH[market], period="max")
    pm = {k: v[v.index >= "2013-01-01"] for k, v in pm.items()}
    bclose = {b: pm.pop(b)["Close"] for b in BENCH[market] if b in pm}
    if market == "tw":
        bclose["SPY"] = bclose["QQQ"] = bclose["0050.TW"]
    F = fx.compute_features(pm, bclose, {s: (theme_of(s).theme if theme_of(s) else None) for s in pm},
                            {s: industry_benchmark(s) for s in pm})
    P = bt.score_panels(F)
    close = F["close"]
    open_px = fx.panel(pm, "Open").reindex(close.index).ffill(limit=3)
    bo = {k: fx._align(bclose[k], close.index) for k in (["SPY", "QQQ"] if market == "us" else ["0050.TW"])}
    top, keep = STRATEGY_PARAMS[market]
    idx = open_px.index[open_px.index >= START]
    rebal_prev = [idx[j - 1] for j in range(1, len(idx) - 1) if (j - 1) % REB == 0]

    v1_w = {"relative_strength": 0.16, "technical": 0.12, "industry_momentum": 0.10, "low_risk": 0.08}
    b_score = bt.opportunity_from(P, penalty=0.0, weights=v1_w, components=tuple(v1_w))
    c_score = bt.opportunity_from(P, penalty=0.6, weights=v1_w, components=tuple(v1_w))
    res = {}
    print(f"[{market}] Model A：重算舊 analyst voting 分數（{len(rebal_prev)} 個調整日）…", flush=True)
    a_score = model_a_scores(pm, open_px, rebal_prev)
    res["A 原本 analyst voting"] = bt.run_strategy(a_score, open_px, top, keep)
    res["B 量化因子"] = bt.run_strategy(b_score, open_px, top, keep)
    res["C 量化 + 相對強度 + 過熱（無 catalyst）"] = bt.run_strategy(c_score, open_px, top, keep)
    mom = bt.run_strategy(P["sharpe_126"], open_px, top, keep, overext=P["overextension"], wait_above=60)
    low = lowentry_open(close, open_px)
    res["D 完整（70% 低檔 + 30% 動能 + 輪動）"] = combine({"low": low, "mom": mom}, {"low": 0.7, "mom": 0.3})
    res["D1 只有低檔布局"] = low
    res["D2 只有動能排名"] = mom
    ew = (1 + (open_px.shift(-1) / open_px - 1).loc[idx].mean(axis=1).fillna(0)).cumprod()
    res["同池等權持有"] = {"equity": ew, "trades": [], "turnover_per_year": 0}

    table = {k: bt.split_metrics(v, bo) for k, v in res.items()}
    for b, s in bo.items():
        e = s.loc[idx]
        table[f"{b} 買進持有"] = bt.split_metrics({"equity": e / e.iloc[0], "trades": [], "turnover_per_year": 0}, bo)

    # Walk-forward：每年用前 3 年 Sharpe 在 A~D 中挑一個，套用到下一年（樣本外串接）
    cands = {k: res[k]["equity"] for k in list(res)[:4]}
    pieces, picks = [], {}
    for y in range(2020, idx[-1].year + 1):
        best, best_sh = None, -9
        for k, e in cands.items():
            rr = e[f"{y - 3}-01-01":f"{y - 1}-12-31"].pct_change().dropna()
            sh = rr.mean() / rr.std() if len(rr) > 60 and rr.std() else -9
            if sh > best_sh:
                best, best_sh = k, sh
        picks[y] = best
        pieces.append(cands[best][f"{y - 1}-12-31":f"{y}-12-31"].pct_change().dropna())
    wf_eq = (1 + pd.concat(pieces)).cumprod()
    wf = {"metrics": bt.metrics(wf_eq, None, bo), "yearly_picks": picks}
    # 各模型的逐年樣本外（每一年都是獨立檢查）
    yearly = {}
    for k in list(res)[:4]:
        e = res[k]["equity"]
        yearly[k] = {str(y): round(float(e[f"{y}-01-01":f"{y}-12-31"].iloc[-1] / e[:f"{y - 1}-12-31"].iloc[-1] - 1) * 100, 1)
                     for y in range(2018, idx[-1].year + 1) if len(e[:f"{y - 1}-12-31"]) and len(e[f"{y}-01-01":f"{y}-12-31"])}
    return {"market": market, "universe_size": int(close.shape[1]), "params": {"top": top, "keep": keep, "rebalance_days": REB},
            "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "execution_timing": "T-1 close signal → T open execution, open-to-open, cost 0.2%/trade",
            "results": table, "walk_forward": wf, "yearly_returns_pct": yearly,
            "notes": ["Model C 無法納入 catalyst：歷史新聞沒有 point-in-time 來源（納入即作弊）。",
                      "Model A 只含舊系統可用歷史價格重算的部分；舊系統的 LLM 人設分析師無法回測（需付費 LLM 與歷史財報）。",
                      "股票池為今天的 AI 科技股（後見之明偏差），所有模型都被高估；請看相對差異。"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=["us", "tw"])
    a = ap.parse_args()
    out = run(a.market)
    with open(os.path.join("docs", "data", f"ablation_{a.market}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("done", a.market)


if __name__ == "__main__":
    main()
