"""持股 加碼 / 續抱 / 減碼 / 出場 判斷（網站「持股健檢」與自動提醒共用同一套規則）。

2026-10-06 改版，與 AI 主線回測（不設固定停利、不設持有期限）一致：
  - 加碼：持股出現「爆量長紅點火」（事件研究：點火後 60 日平均勝同池 +2.5%~5.5%）。
    「上升趨勢中回檔加碼」實測超額為負（-1%~-1.8%），故不以回檔當加碼訊號。
  - 出場：自 60 日收盤高點回落 30%（回測最佳移動停利）或「跌破 MA120 且回落 15% 以上」。
  - 不再因 RSI 過熱而賣出——那等同固定停利，回測證實會砍掉主升段（MU/AMD/DELL）。
  - 仍保留：長線虧損閘門（多重證據）、跌破保護停損且已失守 MA120。
"""
from __future__ import annotations

TREND_EXIT_DRAWDOWN_PCT = 30.0   # 移動停利：自 60 日收盤高點回落 %
MA120_TRIM_DRAWDOWN_PCT = 15.0   # 跌破 MA120 且回落達此 % → 減碼
CONCENTRATION_LIMIT_PCT = 15.0   # 單一持股佔總資產上限（自動提醒用）

ADD = "加碼"
EXIT_RANK = {"停損出場": 3, "獲利了結": 3, "分批減碼": 2, "觀察減碼": 2, ADD: 2, "強勢續抱": 1, "續抱觀察": 1}


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_holding_verdict(report, cost_basis: float | None) -> tuple[str, str, str, str]:
    """回傳 (verdict, urgency, trim_ratio, reason)。trim_ratio 為賣出比例；加碼時為「+N%」。"""
    close = report.latest_close
    ma120 = getattr(report, "ma120", 0.0) or 0.0
    ma20 = getattr(report, "ma20", 0.0) or 0.0
    rsi = report.rsi14
    bias = report.bias
    pnl_pct = None if cost_basis in (None, 0) else ((close / cost_basis) - 1) * 100
    dd60 = _to_float(getattr(report, "drawdown_from_high_pct", None)) or 0.0
    ign_days = getattr(report, "ignition_days_ago", None)

    pf = report.price_forecast if isinstance(getattr(report, "price_forecast", None), dict) else {}
    exp6: float | None = None
    for h in pf.get("horizons", []) or []:
        if h.get("days") == 126:
            exp6 = _to_float(h.get("expected_return_pct"))
            break

    ltr = report.long_term_risk if isinstance(getattr(report, "long_term_risk", None), dict) else {}
    ltr_blocked = bool(ltr.get("blocked"))

    stop_val = None
    try:
        if report.stop_loss and isinstance(report.stop_loss, str):
            stop_val = float(report.stop_loss.split("-")[0].strip())
        elif report.stop_loss:
            stop_val = float(report.stop_loss)
    except Exception:
        stop_val = None
    stop_broken = stop_val is not None and close < stop_val
    below_ma120 = ma120 > 0 and close < ma120

    def _exit(reason_core: str) -> tuple[str, str, str, str]:
        if pnl_pct is not None and pnl_pct < 0:
            return ("停損出場", "高", "100%", "停損出場：" + reason_core)
        return ("獲利了結", "高", "100%", "獲利了結：" + reason_core)

    # ---- 1) 趨勢出場（與回測同一條規則）----
    if dd60 <= -TREND_EXIT_DRAWDOWN_PCT:
        return _exit(
            f"自 60 日高點回落 {abs(dd60):.0f}%，觸發 {TREND_EXIT_DRAWDOWN_PCT:.0f}% 移動停利——趨勢已破，"
            f"依回測紀律全數出場，資金轉往點火/強勢標的。"
        )
    if ltr_blocked:
        return _exit(f"長線虧損閘門觸發（{ltr.get('note', '12 個月統計期望報酬偏負')}），長抱仍難轉正。")
    if stop_broken and below_ma120:
        return _exit(
            f"跌破保護停損 {stop_val:.2f} 且失守長線 MA120（{ma120:.2f}），短中長線同時轉弱，先全數出場。"
        )

    # ---- 2) 加碼：持股爆量長紅點火 ----
    if ign_days is not None and not (ma120 > 0 and close < ma120 * 0.85):
        when = "今天" if ign_days == 0 else f"{ign_days} 天前"
        return (
            ADD, "中", "+20%",
            f"🔥 {getattr(report, 'ignition_note', '') or when + '爆量長紅點火'}。"
            f"回測顯示點火後 1~3 個月平均跑贏同族群，可加碼約原部位兩成；加碼部分停損設點火日低點，"
            f"且單一持股勿超過總資產 {CONCENTRATION_LIMIT_PCT:.0f}%。",
        )

    # ---- 3) 減碼：失守 MA120 且已明顯回落 ----
    if below_ma120 and dd60 <= -MA120_TRIM_DRAWDOWN_PCT:
        return (
            "分批減碼", "中", "50%",
            f"股價跌破長線 MA120（{ma120:.2f}）且自 60 日高點回落 {abs(dd60):.0f}%，趨勢轉弱；"
            f"先減碼一半，若續跌至回落 {TREND_EXIT_DRAWDOWN_PCT:.0f}% 再全數出場，重新站回 MA120 可買回。",
        )
    if stop_broken:
        return (
            "觀察減碼", "中", "30%",
            f"跌破短線保護停損 {stop_val:.2f}，但仍在長線 MA120 之上；先減碼約三成控制風險，核心部位續抱。",
        )

    # ---- 4) 續抱（不因過熱賣出，讓獲利奔跑）----
    hot = f"RSI {rsi:.0f} 偏熱，過熱時勿追加，但不是賣點；" if rsi >= 75 else ""
    if bias == "偏多" and dd60 >= -2.0:
        trail_ref = f"（約 {ma20:.2f}）" if ma20 > 0 else ""
        return (
            "強勢續抱", "低", "0%",
            f"股價貼近 60 日高點、趨勢偏多。{hot}不設固定停利，抱住強勢股；"
            f"跌破 20 日均線{trail_ref}留意，自高點回落 {TREND_EXIT_DRAWDOWN_PCT:.0f}% 才出場。",
        )
    caution = f"　註：股價在長線 MA120（{ma120:.2f}）之下，屬整理期，留意而非賣出訊號。" if below_ma120 else ""
    exp_txt = f"未來半年統計期望報酬 {exp6:.1f}%；" if exp6 is not None else ""
    return (
        "續抱觀察", "低", "0%",
        f"{exp_txt}{hot}無觸發移動停利（目前距 60 日高點 {dd60:.0f}%）、無長線虧損風險，續抱觀察。{caution}",
    )
