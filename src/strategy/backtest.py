"""Sharpe 動能輪動的投組回測（與 src.strategy.momentum 同規則），對標 VOO / 0050。

月調（每 21 個交易日）、持有前 TOP_N[市場]、跌出前 KEEP_N[市場] 才賣、每換一檔扣成本 COST。
以「前一日」的分數決定隔日持股，避免偷看未來。結果分全期與前後兩段（樣本外檢查）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategy.momentum import (ALLOCATION, KEEP_N, LOOKBACK, LOWENTRY_DD, LOWENTRY_LT_YEARS, LOWENTRY_SLOTS,
                                   LIMIT_VALID_DAYS, PANIC_NO_SELL_FG, SPIKE_DAYS, SPIKE_PCT, TECH_BLEND, TOP_N, blend_rank_scores, download_closes, tech_score,
                                   universe)

REBALANCE_DAYS = 21
COST = 0.002
BENCHMARK = {"us": "VOO", "tw": "0050.TW"}
# 額外對照：科技 ETF（美股）＋「同一股票池全部等權持有」——看策略是否只是在吃族群本身的漲幅。
EXTRA_COMPARE = {"us": ["QQQ"], "tw": []}
SPLIT_DATE = "2022-01-01"


def _stats(eq: pd.Series) -> Dict:
    eq = eq / eq.iloc[0]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dr = eq.pct_change().dropna()
    return {
        "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
        "sharpe": round(float(dr.mean() / dr.std() * math.sqrt(252)), 2) if dr.std() else 0.0,
    }


def run_backtest(market: str, closes: pd.DataFrame, bench: pd.Series, start: str = "2017-07-01",
                 extra: Optional[Dict[str, pd.Series]] = None, opens: Optional[pd.DataFrame] = None,
                 fg: Optional[pd.Series] = None, tech: Optional[Dict] = None, no_chase: bool = True) -> Dict:
    """時序（point-in-time）：T-1 收盤算分數 → T 開盤成交 → 持有報酬以開盤對開盤計算。
    2026-10-07 修正：舊版用 T-1 收盤的分數、又從同一個 T-1 收盤開始計報酬（等於看完收盤價再用收盤價成交）。
    未提供 opens 時退而用「T 收盤成交」（訊號後下一個收盤，仍不使用訊號當日的價格成交）。
    fg（恐懼貪婪，已對齊為「T-1 收盤可知」）：月調日 T-1 值 < PANIC_NO_SELL_FG → 只買不賣。
    tech：{T-1 日期: 各股 virattt 技術分數}（只用 T-1 以前的 K 線），依 TECH_BLEND[market] 混入排名。
    no_chase：月調要新買的股票若 T-1 以前 SPIKE_DAYS 日內有單日漲 ≥ SPIKE_PCT → 先不買，之後每天檢查，
    大長紅超過 SPIKE_DAYS 日且仍在前 TOP_N 名才買（到下次月調為止）。"""
    closes = closes.sort_index().ffill(limit=3)
    rets = closes.pct_change()
    if opens is not None:
        opens = opens.reindex(closes.index).ffill(limit=3)
        hold_ret = (opens.shift(-1) / opens - 1)          # 第 i 天開盤買進，持有到第 i+1 天開盤
    else:
        hold_ret = rets.shift(-1)                          # 第 i 天收盤買進，賺第 i+1 天的報酬
    score = rets.rolling(LOOKBACK).mean() / rets.rolling(LOOKBACK).std()
    spiky = (rets.rolling(SPIKE_DAYS, min_periods=1).max() >= SPIKE_PCT)
    idx = closes.index[closes.index >= start]
    spiky = spiky.loc[idx]
    closes, rets, score = closes.loc[idx], rets.loc[idx].fillna(0.0), score.loc[idx]
    hold_ret = hold_ret.loc[idx].fillna(0.0)
    bench = bench.reindex(idx).ffill()
    if fg is not None:
        fg = fg.reindex(idx)

    held: List[str] = []
    pending: List[str] = []
    t_last = None
    eq = [1.0]
    trades = 0
    log: List[Dict] = []
    for i in range(1, len(idx)):
        cost = 0.0
        if (i - 1) % REBALANCE_DAYS == 0:
            w = TECH_BLEND.get(market, 0.0)
            t_prev = tech.get(idx[i - 1]) if (tech and w) else None
            t_last = t_prev
            s = blend_rank_scores(score.iloc[i - 1], t_prev, w if t_prev is not None else 0.0).sort_values(ascending=False)
            pos = {k: j for j, k in enumerate(s.index)}
            f_prev = float(fg.iloc[i - 1]) if fg is not None and np.isfinite(fg.iloc[i - 1]) else None
            if f_prev is not None and f_prev < PANIC_NO_SELL_FG:
                stay = list(held)                                  # 極度恐懼：只買不賣
            else:
                stay = [h for h in held if pos.get(h, 10 ** 9) < KEEP_N[market]]
            new = [k for k in s.index if k not in stay][: max(0, TOP_N[market] - len(stay))]
            pending = [k for k in new if no_chase and bool(spiky.iloc[i - 1].get(k, False))]
            nxt = stay + [k for k in new if k not in pending]
            changed = set(nxt) ^ set(held)
            trades += len(changed)
            cost = COST * len(changed) / TOP_N[market]
            if changed:
                log.append({"date": idx[i].strftime("%Y-%m-%d"),
                            "buy": sorted(set(nxt) - set(held)), "sell": sorted(set(held) - set(nxt))})
            held = nxt
        elif pending:                                              # 等大長紅過後再買
            w = TECH_BLEND.get(market, 0.0)
            s = blend_rank_scores(score.iloc[i - 1], t_last, w if t_last is not None else 0.0).sort_values(ascending=False)
            top = set(s.index[:TOP_N[market]])
            buy = [k for k in pending if k in top and not bool(spiky.iloc[i - 1].get(k, False))][: max(0, TOP_N[market] - len(held))]
            pending = [k for k in pending if k not in buy and k in top]
            if buy:
                held = held + buy
                trades += len(buy)
                cost = COST * len(buy) / TOP_N[market]
                log.append({"date": idx[i].strftime("%Y-%m-%d"), "buy": sorted(buy), "sell": []})
        day = float(hold_ret.iloc[i][held].mean()) if held else 0.0
        eq.append(eq[-1] * (1 + day - cost))
    eq_s = pd.Series(eq, index=idx)
    ew = (1 + rets.mean(axis=1).fillna(0.0)).cumprod()   # 同池等權（每日再平衡近似）
    bm = bench / bench.iloc[0]

    periods = {"全期": (None, None), "2017-2021": (None, "2021-12-31"), "2022-今": (SPLIT_DATE, None)}
    out_periods = {}
    for lab, (a, z) in periods.items():
        e, b = eq_s[a:z], bm[a:z]
        if len(e) > 30:
            out_periods[lab] = {"strategy": _stats(e), "benchmark": _stats(b)}
            comp = {"同池等權持有": _stats(ew[a:z])}
            for name, ser in (extra or {}).items():
                comp[name] = _stats(ser.reindex(idx).ffill()[a:z])
            out_periods[lab]["compare"] = comp

    weekly = eq_s.iloc[::5]
    bm_weekly = bm.reindex(weekly.index)
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "market": market,
        "benchmark_symbol": BENCHMARK[market],
        "start_date": idx[0].strftime("%Y-%m-%d"),
        "end_date": idx[-1].strftime("%Y-%m-%d"),
        "universe_size": int(closes.shape[1]),
        "trades_per_year": round(trades / years, 1),
        "periods": out_periods,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "strategy": round(float(v), 4), "benchmark": round(float(bm_weekly.loc[d]), 4)}
            for d, v in weekly.items()
        ],
        "_eq_daily": eq_s,
        "current_holdings": held,
        "recent_rebalances": log[-6:],
        "rules": {"lookback_days": LOOKBACK, "top_n": TOP_N[market], "keep_n": KEEP_N[market],
                  "rebalance_days": REBALANCE_DAYS, "cost_per_trade": COST},
        "caveat": "AI 科技股池是用現在眼光挑出的贏家族群，有明顯倖存者／後見之明偏差：實際報酬會比回測低很多，AI 族群轉弱時也可能大幅落後大盤。",
    }


def lowentry_returns(closes: pd.DataFrame, start: str = "2017-07-01", hold: int = 252,
                     opens: Optional[pd.DataFrame] = None, idle_ret: Optional[pd.Series] = None,
                     lows: Optional[pd.DataFrame] = None, no_chase: bool = True) -> Tuple[pd.Series, List[Dict]]:
    """長線低檔布局：T-1 收盤符合（3 年報酬 > 0 且距 52 週高點 ≤ -30%）→ T 成交，持有 hold 日。
    提供 opens 時：T 開盤成交、開盤對開盤計報酬（與動能回測同一時間基準，組合時才不會虛增分散效果）；
    否則 T 收盤成交、收盤對收盤。每檔 1/LOWENTRY_SLOTS 權重；同檔出場後 60 日內不重複進場。
    idle_ret（與本函式同一時間基準的日報酬）：空槽資金放在這裡（動能名單），進出低檔時多付一次換股成本；None = 空槽為現金。
    no_chase（需 opens 與 lows）：T-1 以前 SPIKE_DAYS 日內有單日漲 ≥ SPIKE_PCT → 不在 T 開盤買，改掛「大漲前一日收盤」限價，
    之後 LIMIT_VALID_DAYS 日內盤中低點觸及即以 min(開盤, 限價) 成交；逾時放棄、20 日內不再掛。限價單佔用槽位。"""
    closes = closes.sort_index().ffill(limit=3)
    if opens is not None:
        px_exec = opens.reindex(closes.index).ffill(limit=3)
        rets = (px_exec / px_exec.shift(1) - 1)            # 第 j 天的報酬 = 開盤(j-1) → 開盤(j)
    else:
        px_exec = closes
        rets = closes.pct_change()
    dd = closes / closes.rolling(252, min_periods=200).max() - 1
    lt = closes / closes.shift(252 * LOWENTRY_LT_YEARS) - 1
    ok = (dd <= LOWENTRY_DD) & (lt > 0)
    use_limit = no_chase and opens is not None and lows is not None
    if use_limit:
        lows = lows.reindex(closes.index).ffill(limit=3)
        cr = closes.pct_change()
        spiky = cr.rolling(SPIKE_DAYS, min_periods=1).max() >= SPIKE_PCT
    orders: Dict[str, Tuple[float, int]] = {}
    fresh: Dict[str, float] = {}
    idx = closes.index[closes.index >= start]
    pos: Dict[str, Tuple[int, float]] = {}
    cool: Dict[str, int] = {}
    daily, trades = [], []
    for j in range(1, len(idx)):
        d, prev = idx[j], idx[j - 1]
        r = 0.0
        for k in pos:
            v = rets.at[d, k]
            if k in fresh and pos[k][0] == j - 1:          # 昨天以限價成交：報酬從成交價起算
                v = px_exec.at[d, k] / fresh.pop(k) - 1
            if np.isfinite(v):
                r += float(v)
        r /= LOWENTRY_SLOTS
        if idle_ret is not None:
            ir = idle_ret.get(d, 0.0)
            r += (LOWENTRY_SLOTS - len(pos)) / LOWENTRY_SLOTS * (float(ir) if np.isfinite(ir) else 0.0)
        switch = 2 if idle_ret is not None else 1
        cost = 0.0
        for k in list(pos):
            j0, p0 = pos[k]
            if j - j0 >= hold:
                trades.append({"ticker": k, "entry": idx[j0].strftime("%Y-%m-%d"), "exit": d.strftime("%Y-%m-%d"),
                               "return_pct": round((px_exec.at[d, k] / p0 - 1) * 100, 1)})
                del pos[k]
                cool[k] = j + 60
                cost += switch * COST / LOWENTRY_SLOTS
        for k in list(orders):                             # 限價單：今天盤中低點觸及即成交
            lim, exp = orders[k]
            if np.isfinite(lows.at[d, k]) and lows.at[d, k] <= lim and np.isfinite(px_exec.at[d, k]):
                px = min(float(px_exec.at[d, k]), lim)
                pos[k] = (j, px)
                if px != float(px_exec.at[d, k]):
                    fresh[k] = px
                cost += switch * COST / LOWENTRY_SLOTS
                del orders[k]
            elif j >= exp:
                del orders[k]
                cool[k] = j + 20
        if len(pos) + len(orders) < LOWENTRY_SLOTS:
            row = ok.loc[prev]
            cands = sorted([k for k in row[row].index if k not in pos and k not in orders and cool.get(k, 0) <= j],
                           key=lambda k: dd.at[prev, k])
            for k in cands[: LOWENTRY_SLOTS - len(pos) - len(orders)]:
                if not np.isfinite(px_exec.at[d, k]):
                    continue
                if use_limit and bool(spiky.at[prev, k]):
                    hist = cr[k].loc[:prev].tail(SPIKE_DAYS)
                    b = hist.index[(hist >= SPIKE_PCT).values][0]
                    lim = closes[k].shift(1).at[b]
                    if np.isfinite(lim):
                        orders[k] = (float(lim), j + LIMIT_VALID_DAYS)
                    continue
                pos[k] = (j, float(px_exec.at[d, k]))
                cost += switch * COST / LOWENTRY_SLOTS
        daily.append((d, r - cost))
    ser = pd.Series([x for _, x in daily], index=[d for d, _ in daily])
    return ser, trades


def _period_stats(eq: pd.Series, bench: pd.Series) -> Dict:
    out = {}
    for lab, (a, z) in {"全期": (None, None), "2017-2021": (None, "2021-12-31"), "2022-今": (SPLIT_DATE, None)}.items():
        e, b = eq[a:z], bench[a:z]
        if len(e) > 30:
            out[lab] = {"strategy": _stats(e), "benchmark": _stats(b)}
    return out


def _fg_for(index: pd.Index, market: str, fg_all: Optional[pd.Series]) -> Optional[pd.Series]:
    """恐懼貪婪對齊到市場交易日，值 = 該日收盤後可知。台股保守再延一個交易日（美股收盤晚於台股）。"""
    if fg_all is None:
        return None
    idx = pd.to_datetime(index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    s = fg_all.reindex(idx, method="ffill")
    if market == "tw":
        s = s.shift(1)
    s.index = index
    return s


def tech_scores_at_rebalances(pm: Dict[str, pd.DataFrame], index: pd.Index, start: str = "2017-07-01") -> Dict:
    """回測用：每個月調訊號日（T-1）各股 virattt 技術分數，只用當日以前的 K 線。"""
    idx = index[index >= start]
    out = {}
    for i in range(1, len(idx)):
        if (i - 1) % REBALANCE_DAYS:
            continue
        p = idx[i - 1]
        sc = {}
        for k, f in pm.items():
            v = tech_score(f.loc[:p])
            if v is not None:
                sc[k] = v
        out[p] = pd.Series(sc, dtype=float)
    return out


def _sentiment_ablation(m, closes, bench, opens, fg, combo_eq, b, tech=None, lows=None) -> List[Dict]:
    """同一資料比較：原規則 → 逐步加上 空槽放動能 / 極度恐懼只買不賣 / virattt 技術分數（台股）/ 不追大長紅（目前規則）。"""
    rows = []
    try:
        variants = []
        steps = [("原規則（空槽現金、恐慌照賣）", False, False, False), ("＋ 空槽放動能", True, False, False),
                 ("＋ 極度恐懼只買不賣", True, True, False)]
        if tech:
            steps.append(("＋ virattt 技術分數混入排名", True, True, True))
        for name, use_idle, use_fg, use_tech in steps:
            r = run_backtest(m, closes, bench, opens=opens, fg=fg if use_fg else None, tech=tech if use_tech else None,
                             no_chase=False)
            e = r["_eq_daily"]
            e.index = pd.to_datetime(e.index)
            mr = e.pct_change().shift(1)
            lr, _ = lowentry_returns(closes, opens=opens, idle_ret=mr if use_idle else None, no_chase=False)
            cr = ALLOCATION["lowentry"] * lr + ALLOCATION["momentum"] * mr.reindex(lr.index).fillna(0)
            variants.append((name, (1 + cr).cumprod()))
        variants.append(("＋ 不追大長紅（目前規則）", combo_eq))
        for name, eq in variants:
            ps = _period_stats(eq, b)
            rows.append({"name": name, **{lab: ps[lab]["strategy"] for lab in ps}})
    except Exception as e:
        print(f"[backtest] 情緒消融失敗：{e}")
    return rows


def _sentiment_study(closes: pd.DataFrame, bench: pd.Series, fg: Optional[pd.Series]) -> Optional[Dict]:
    """恐懼貪婪分區 → 股票池等權 / 大盤 未來 1、3、6 個月平均報酬（2009 起；只描述歷史，不是預測）。"""
    if fg is None:
        return None
    cl = closes.copy()
    cl.index = pd.to_datetime(cl.index)
    cl.index = cl.index.tz_localize(None) if cl.index.tz is not None else cl.index
    ew = (1 + cl.ffill(limit=3).pct_change().mean(axis=1).fillna(0)).cumprod()
    f = fg.copy()
    f.index = cl.index
    bm = bench.copy()
    bm.index = pd.to_datetime(bm.index)
    bm.index = bm.index.tz_localize(None) if bm.index.tz is not None else bm.index
    bm = bm.reindex(cl.index).ffill()
    df = pd.DataFrame({"fg": f})
    for lab, ser in (("pool", ew), ("bench", bm)):
        for h in (21, 63, 126):
            df[f"{lab}_{h}"] = (ser.shift(-h) / ser - 1) * 100
    df = df[df.index >= "2009-01-01"].dropna(subset=["fg"])
    bins = [(0, 25, "極度恐懼 <25"), (25, 45, "恐懼 25-45"), (45, 55, "中性 45-55"), (55, 75, "貪婪 55-75"), (75, 101, "極度貪婪 >75")]
    rows = []
    for lo, hi, lab in bins:
        sub = df[(df.fg >= lo) & (df.fg < hi)]
        if len(sub) < 30:
            continue
        rows.append({"bucket": lab, "lo": lo, "hi": hi, "days": int(len(sub)),
                     **{k: round(float(sub[k].mean()), 1) for k in sub.columns if k != "fg" and sub[k].notna().sum() > 20}})
    return {"since": "2009-01-01", "rows": rows,
            "note": "恐懼時未來報酬較高（逆向）、貪婪時並沒有明顯較差：所以情緒只用來「恐慌時不賣」，不用來「貪婪時賣出/等待」。"}


def build_backtest_report(period: str = "max") -> Dict:
    out = {}
    fg_all = None
    try:
        from src.strategy.sentiment import SENTIMENT_TICKERS, fear_greed_proxy
        spm = download_closes(SENTIMENT_TICKERS, period=period)
        fg_all = fear_greed_proxy({k: f["Close"] for k, f in spm.items()})["fg"]
    except Exception as e:
        print(f"[backtest] 恐懼貪婪歷史計算失敗，回測不含情緒規則：{e}")
    for m in ("us", "tw"):
        syms = universe(m)
        pm = download_closes(syms + [BENCHMARK[m]], period=period)
        extra_syms = EXTRA_COMPARE[m]
        if extra_syms:
            pm.update(download_closes(extra_syms, period=period))
        extra = {e: pm.pop(e)["Close"] for e in extra_syms if e in pm}
        bench = pm.pop(BENCHMARK[m], None)
        if bench is None or not pm:
            continue
        closes = pd.DataFrame({s: f["Close"] for s, f in pm.items()})
        closes = closes[closes.index >= "2013-01-01"]
        opens = pd.DataFrame({s: f["Open"] for s, f in pm.items() if "Open" in f}).reindex(closes.index)
        lows = pd.DataFrame({s: f["Low"] for s, f in pm.items() if "Low" in f}).reindex(closes.index)
        fg = _fg_for(closes.index, m, fg_all)
        tech = None
        if TECH_BLEND.get(m):
            print(f"[backtest] {m}：計算 virattt 技術分數（每個月調日）…", flush=True)
            tech = tech_scores_at_rebalances({s: f[f.index >= "2013-01-01"] for s, f in pm.items()}, closes.index)
        res = run_backtest(m, closes, bench["Close"], extra=extra, opens=opens, fg=fg, tech=tech)
        res["execution_timing"] = "兩條策略皆為 T-1 收盤訊號 → T 開盤成交，開盤對開盤計報酬；組合時兩者對齊同一時間區段"
        eq_m = res.pop("_eq_daily")   # 必須用「每日」動能報酬組合，週取樣會讓波動/Sharpe 失真
        eq_m.index = pd.to_datetime(eq_m.index)
        # 動能 eq_m 第 i 筆 = 開盤(i) → 開盤(i+1)；低檔 low_r 第 j 筆 = 開盤(j-1) → 開盤(j)：動能往後平移一格（shift(1)）對齊同一段時間
        mom_r_all = eq_m.pct_change().shift(1)
        # 長線低檔布局：空槽資金放動能（IDLE_TO_MOMENTUM）
        low_r, low_trades = lowentry_returns(closes, opens=opens, idle_ret=mom_r_all, lows=lows)
        mom_r = mom_r_all.reindex(low_r.index).fillna(0)
        combo_r = ALLOCATION["lowentry"] * low_r + ALLOCATION["momentum"] * mom_r
        low_eq, combo_eq = (1 + low_r).cumprod(), (1 + combo_r).cumprod()
        b = bench["Close"].copy()
        b.index = pd.to_datetime(b.index).tz_localize(None) if pd.to_datetime(b.index).tz is not None else pd.to_datetime(b.index)
        b = b.reindex(low_r.index).ffill()
        res["tracks"] = {
            "lowentry": {"name": "長線低檔布局（空槽放動能）", "periods": _period_stats(low_eq, b),
                         "recent_trades": low_trades[-8:], "trades": len(low_trades),
                         "win_rate_pct": round(sum(1 for t in low_trades if t["return_pct"] > 0) / max(len(low_trades), 1) * 100, 1)},
            "combo": {"name": f"組合：低檔 {ALLOCATION['lowentry']:.0%} ＋ 動能 {ALLOCATION['momentum']:.0%}（含情緒規則）",
                      "periods": _period_stats(combo_eq, b)},
        }
        res["sentiment_ablation"] = _sentiment_ablation(m, closes, bench["Close"], opens, fg, combo_eq, b, tech, lows)
        res["sentiment_study"] = _sentiment_study(closes, b, fg)
        weekly = combo_eq.iloc[::5]
        low_w = low_eq.reindex(weekly.index)
        res["combo_curve"] = [{"date": d.strftime("%Y-%m-%d"), "combo": round(float(v), 4), "lowentry": round(float(low_w.loc[d]), 4),
                               "benchmark": round(float(b.loc[d] / b.iloc[0]), 4)} for d, v in weekly.items()]
        out[m] = res
    from datetime import datetime, timedelta, timezone
    return {"generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "allocation": ALLOCATION, "markets": out}
