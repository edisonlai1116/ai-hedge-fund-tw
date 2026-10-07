"""價格／量能特徵（point-in-time）。

所有特徵都以「日期 × 股票」的面板計算，第 t 列只用到 t（含）以前的資料。
回測在 T 日下單時只取 T-1 那一列；即時評分取最後一列。兩者共用同一份程式，結果一致。

報酬一律用複利（價格比），不是日報酬加總。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

HORIZONS = (20, 60, 120, 252)
RS_WEIGHTS = {20: 0.20, 60: 0.30, 120: 0.30, 252: 0.20}


def panel(price_map: Dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    cols = {}
    for s, f in price_map.items():
        if f is None or f.empty or field not in f.columns:
            continue
        ser = f[field].copy()
        idx = pd.to_datetime(ser.index)
        ser.index = idx.tz_localize(None) if idx.tz is not None else idx
        cols[s] = ser[~ser.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _ramp(x, lo: float, hi: float):
    """lo→0、hi→100 的線性映射（夾在 0~100）。lo > hi 時方向相反。"""
    return ((x - lo) / (hi - lo) * 100).clip(0, 100)


def compute_features(
    price_map: Dict[str, pd.DataFrame],
    bench_close: Dict[str, pd.Series],
    theme_of: Optional[Dict[str, str]] = None,
    industry_etf: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """price_map: {sym: OHLCV DataFrame（欄位 Open/High/Low/Close/Volume）}。
    bench_close: {"SPY": Series, "QQQ": Series, "SMH": ..., ...}
    回傳 {feature_name: DataFrame(dates × symbols)}。"""
    close = panel(price_map, "Close").ffill(limit=3)
    high = panel(price_map, "High").reindex_like(close)
    low = panel(price_map, "Low").reindex_like(close)
    vol = panel(price_map, "Volume").reindex_like(close)
    idx = close.index
    bench = {k: _align(v, idx) for k, v in bench_close.items() if v is not None}
    rets = close.pct_change()
    F: Dict[str, pd.DataFrame] = {"close": close}

    # 複利報酬
    for n in (5, 50) + HORIZONS:
        F[f"ret_{n}"] = close / close.shift(n) - 1

    # 相對報酬（vs SPY、QQQ、產業 ETF、同主題平均）
    for n in HORIZONS:
        for b in ("SPY", "QQQ"):
            if b in bench:
                br = bench[b] / bench[b].shift(n) - 1
                F[f"rel_{b}_{n}"] = F[f"ret_{n}"].sub(br, axis=0)
        if industry_etf:
            ind = pd.DataFrame({s: (bench[e] / bench[e].shift(n) - 1) if e in bench else np.nan
                                for s, e in industry_etf.items() if s in close.columns}, index=idx)
            F[f"rel_ind_{n}"] = F[f"ret_{n}"] - ind.reindex(columns=close.columns)
        if theme_of:
            groups: Dict[str, list] = {}
            for s in close.columns:
                if theme_of.get(s):
                    groups.setdefault(theme_of[s], []).append(s)
            peer = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
            for members in groups.values():
                if len(members) >= 2:
                    tot = F[f"ret_{n}"][members].sum(axis=1)
                    cnt = F[f"ret_{n}"][members].notna().sum(axis=1)
                    for s in members:   # 排除自己的同主題平均
                        peer[s] = (tot - F[f"ret_{n}"][s].fillna(0)) / (cnt - F[f"ret_{n}"][s].notna()).replace(0, np.nan)
            F[f"rel_theme_{n}"] = F[f"ret_{n}"] - peer
            F[f"theme_ret_{n}"] = peer

    # 均線、52 週、RSI、量能
    ma20, ma50, ma200 = close.rolling(20).mean(), close.rolling(50).mean(), close.rolling(200).mean()
    F["dist_ma20"] = close / ma20 - 1
    F["dist_ma50"] = close / ma50 - 1
    F["dist_ma200"] = close / ma200 - 1
    F["ma50_gt_ma200"] = (ma50 > ma200).astype(float).where(ma200.notna())
    F["ma50_slope20"] = ma50 / ma50.shift(20) - 1
    hi252 = close.rolling(252, min_periods=120).max()
    lo252 = close.rolling(252, min_periods=120).min()
    F["dist_52w_high"] = close / hi252 - 1
    F["pos_52w_range"] = (close - lo252) / (hi252 - lo252).replace(0, np.nan)
    F["rsi14"] = _rsi(close)
    vol20 = vol.shift(1).rolling(20).mean()
    F["vol_spike"] = vol / vol20
    up_vol = vol.where(rets > 0, 0).rolling(20).sum()
    F["up_volume_ratio"] = up_vol / vol.rolling(20).sum().replace(0, np.nan)
    F["breakout"] = ((close >= hi252 * 0.99) & (F["vol_spike"] >= 1.3)).astype(float).where(hi252.notna())

    # 波動、ATR、beta、回撤
    F["vol_63"] = rets.rolling(63).std() * math.sqrt(252)
    F["daily_vol_63"] = rets.rolling(63).std()
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).stack(), (high - prev_close).abs().stack(), (low - prev_close).abs().stack()], axis=1).max(axis=1).unstack()
    F["atr14"] = tr.reindex_like(close).rolling(14).mean()
    if "SPY" in bench:
        mret = bench["SPY"].pct_change()
        cov = rets.rolling(252, min_periods=120).cov(mret)
        F["beta_252"] = cov.div(mret.rolling(252, min_periods=120).var(), axis=0)
        F["spy_ret_1"] = pd.DataFrame({s: mret for s in close.columns}, index=idx)
    F["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1
    F["ret_1"] = rets
    F["max_gain_5"] = rets.rolling(5).max()          # 近 5 日最大單日漲幅（跳空／加速）

    # 長線低檔布局（已驗證）：3 年報酬（長線贏家判定）
    F["ret_756"] = close / close.shift(756) - 1

    # 已驗證的核心因子：6 個月風險調整動能
    F["sharpe_126"] = rets.rolling(126).mean() / rets.rolling(126).std()
    return F


def _align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = s.copy()
    i = pd.to_datetime(s.index)
    s.index = i.tz_localize(None) if i.tz is not None else i
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(idx).ffill(limit=3)


# ---------------------------------------------------------------------------
# 分數（0~100）。相對分數用「當日橫斷面百分位」；過熱/風險用「有經濟意義的絕對門檻」。
# ---------------------------------------------------------------------------
def pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) * 100


def relative_strength_score(F: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts, wsum = [], 0.0
    for n, w in RS_WEIGHTS.items():
        comps = [pct_rank(F[k]) for k in (f"rel_SPY_{n}", f"rel_QQQ_{n}", f"rel_ind_{n}", f"rel_theme_{n}") if k in F]
        if comps:
            parts.append(sum(c.fillna(50) for c in comps) / len(comps) * w)
            wsum += w
    out = sum(parts) / wsum
    return out.where(F["ret_252"].notna() | F["ret_120"].notna())


def technical_score(F: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """趨勢（均線結構）+ 動能（6 個月風險調整動能百分位）。波動不給方向。"""
    trend = (
        (F["dist_ma50"] > 0).astype(float) * 25
        + (F["dist_ma200"] > 0).astype(float) * 25
        + F["ma50_gt_ma200"].fillna(0) * 25
        + (F["ma50_slope20"] > 0).astype(float) * 25
    ).where(F["dist_ma200"].notna())
    momentum = pct_rank(F["sharpe_126"])
    return (0.5 * trend + 0.5 * momentum).where(momentum.notna())


# 過熱分層（0~100）
OVEREXTENSION_TIERS = ((85, "Extreme"), (70, "Overheated"), (50, "Extended"), (25, "Warm"), (0, "Normal"))


# 原始分數 → 校準分數的錨點：美股 AI 池 + 台股 AI 池 2015-01 ~ 2026-10 全部（股票 × 日）共 417,981 筆的原始分數分位數
#   P75 = 12.8、P90 = 30.6、P96 = 54.5、P99 = 89.3（依此映射；非針對任何個股調整）。
OVEREXTENSION_CALIBRATION = ((0.0, 0.0), (12.8, 25.0), (30.6, 50.0), (54.5, 70.0), (89.3, 85.0), (100.0, 100.0))


def overextension_tier(score: Optional[float]) -> Optional[str]:
    if score is None or not np.isfinite(score):
        return None
    return next(name for lo, name in OVEREXTENSION_TIERS if score >= lo)


def overextension_parts(F: Dict[str, pd.DataFrame]) -> Dict[str, tuple]:
    """各成分 0~100 與權重。漲幅一律用「絕對漲幅」：舊版以波動單位衡量，高波動股同樣的急漲反而被判成不熱
    （例：20 日 +26%、RSI 69 只得 34 分）；同樣的漲幅，低波動股也不能比高波動股更「熱」。"""
    parts = {
        "ret_5": (_ramp(F["ret_5"], 0.03, 0.15), 0.10),
        "ret_20": (_ramp(F["ret_20"], 0.08, 0.35), 0.18),
        "ret_50": (_ramp(F["ret_50"], 0.15, 0.60), 0.10),
        "rsi": (_ramp(F["rsi14"], 55, 80), 0.15),
        "ma20": (_ramp(F["dist_ma20"], 0.04, 0.20), 0.12),
        "ma50": (_ramp(F["dist_ma50"], 0.08, 0.35), 0.10),
        "ma200": (_ramp(F["dist_ma200"], 0.20, 0.80), 0.08),
        "near_high": (_ramp(F["dist_52w_high"], -0.10, 0.0), 0.05),
        "gap": (_ramp(F["max_gain_5"], 0.04, 0.12), 0.05),
    }
    if "rel_SPY_20" in F:
        parts["rel_strength"] = (_ramp(F["rel_SPY_20"], 0.05, 0.25), 0.07)
    return parts


def overextension_score(F: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """短線過熱（0~100，高=過熱）。多成分（5/20/50 日報酬、RSI、距 20/50/200 日線、距 52 週高、
    相對強度、跳空加速），漲幅用絕對漲幅；上漲中的高波動股再放大（vol 0.4 → ×1.0、0.8 以上 → ×1.2），
    高波動只會讓「快速上漲」更容易被判過熱，不會降低過熱分數。
    分層：0–24 Normal、25–49 Warm、50–69 Extended、70–84 Overheated、85–100 Extreme。"""
    parts = overextension_parts(F)
    num = sum(p.fillna(0) * w for p, w in parts.values())
    den = sum(p.notna().astype(float) * w for p, w in parts.values()).replace(0, np.nan)
    base = num / den
    amp = 1 + 0.2 * _ramp(F["vol_63"], 0.40, 0.80) / 100 * (F["ret_20"] > 0).astype(float)
    raw = (base * amp).clip(0, 100)
    # 校準：把原始分數依「歷史橫斷面百分位」映射到分層（P75→25 Warm、P90→50 Extended、P96→70 Overheated、P99→85 Extreme）
    xp, fp = zip(*OVEREXTENSION_CALIBRATION)
    cal = pd.DataFrame(np.interp(raw.to_numpy(dtype=float), xp, fp), index=raw.index, columns=raw.columns)
    return cal.where(raw.notna() & F["rsi14"].notna())


def risk_score(F: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """風險（0~100，高=風險高）：波動、beta、近一年回撤的絕對門檻。"""
    v = _ramp(F["vol_63"], 0.20, 0.90)
    b = _ramp(F["beta_252"], 0.6, 2.5) if "beta_252" in F else v * np.nan
    dd = _ramp(-F["drawdown_252"], 0.10, 0.50)
    num = v.fillna(0) * 0.45 + b.fillna(0) * 0.25 + dd.fillna(0) * 0.30
    den = v.notna() * 0.45 + b.notna() * 0.25 + dd.notna() * 0.30
    return (num / den.replace(0, np.nan)).where(F["vol_63"].notna())


def industry_momentum_score(F: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """同主題（排除自己）的 60/120 日報酬，對全體做百分位。沒有同主題資料 → NaN（不當中性）。"""
    if "theme_ret_60" not in F:
        return F["close"] * np.nan
    blend = 0.5 * F["theme_ret_60"] + 0.5 * F["theme_ret_120"]
    return pct_rank(blend)


def last_row(df: pd.DataFrame, asof: Optional[pd.Timestamp] = None) -> pd.Series:
    if asof is not None:
        df = df.loc[:asof]
    return df.iloc[-1] if len(df) else pd.Series(dtype=float)
