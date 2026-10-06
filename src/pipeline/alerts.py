"""自動買賣提醒（GitHub Actions 排程呼叫；免 API Key）。

做兩件事，條件成立才推播，同一訊號 5 天內不重複：
  1) 持股提醒：讀你的持股（GitHub Secret `HOLDINGS`，或本機 股票成本.txt），用
     src.holding_rules（與網站「持股健檢」同一套規則）判斷 加碼 / 減碼 / 出場，
     再加上「單一持股 > 15% 總資產」集中度提醒。
  2) 買點提醒：觀察池（台美 AI 主線 + 權值股）出現「爆量長紅點火」→ 推播進場價與停損。

推播管道（設定哪個就用哪個，可多選；全沒設定就只在本機印出 = 試跑）：
  - ntfy：        NTFY_TOPIC（可選 NTFY_SERVER，預設 https://ntfy.sh）
  - Telegram：    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Discord：     DISCORD_WEBHOOK_URL
  - Email(SMTP)： SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / ALERT_EMAIL_TO

隱私：repo 為公開，Actions 日誌任何人都看得到——在 CI（GITHUB_ACTIONS=true）中
**不印出任何持股代號、數量或訊息內容**，只印筆數；持股只從 Secret 讀取。

用法：python -m src.pipeline.alerts [--market auto|us|tw|all] [--dry-run]
  auto：依現在哪個市場開盤決定（盤中排程用）；收盤後的每日排程用 all。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLDINGS_TXT = os.path.join(REPO_ROOT, "股票成本.txt")
STATE_PATH = os.environ.get("ALERT_STATE_PATH", os.path.join(REPO_ROOT, ".alert_state", "state.json"))
DEDUP_DAYS = 5
MAX_BUY_ALERTS = 8
IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"

TPE = timezone(timedelta(hours=8))


# ===== 持股解析 =============================================================
def parse_holdings(text: str) -> List[Dict]:
    """每行 `代號 成本 股數`（與 股票成本.txt 相同格式），# 開頭為註解。同代號合併（加權平均成本）。"""
    merged: Dict[str, Dict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        try:
            cost, shares = float(parts[1].replace(",", "")), float(parts[2].replace(",", ""))
        except ValueError:
            continue
        sym = _yf_symbol(parts[0])
        if sym in merged:
            m = merged[sym]
            total = m["shares"] + shares
            m["cost"] = (m["cost"] * m["shares"] + cost * shares) / total if total else cost
            m["shares"] = total
        else:
            merged[sym] = {"symbol": sym, "cost": cost, "shares": shares}
    return list(merged.values())


def _yf_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." in t:
        return t
    return f"{t}.TW" if t[:1].isdigit() else t


def _is_tw(sym: str) -> bool:
    return sym.endswith((".TW", ".TWO"))


def load_holdings() -> List[Dict]:
    text = os.environ.get("HOLDINGS", "")
    if not text.strip() and os.path.exists(HOLDINGS_TXT):
        with open(HOLDINGS_TXT, encoding="utf-8") as f:
            text = f.read()
    return parse_holdings(text)


# ===== 盤中量能換算 =========================================================
def session_fraction(market: str, now_utc: Optional[datetime] = None) -> Optional[float]:
    """回傳該市場今日盤中已經過的比例（0~1）；未開盤或已收盤回 None。"""
    from zoneinfo import ZoneInfo
    now_utc = now_utc or datetime.now(timezone.utc)
    if market == "us":
        local = now_utc.astimezone(ZoneInfo("America/New_York"))
        start, minutes = local.replace(hour=9, minute=30, second=0, microsecond=0), 390
    else:
        local = now_utc.astimezone(ZoneInfo("Asia/Taipei"))
        start, minutes = local.replace(hour=9, minute=0, second=0, microsecond=0), 270
    if local.weekday() >= 5:
        return None
    elapsed = (local - start).total_seconds() / 60
    if elapsed <= 0 or elapsed >= minutes:
        return None
    return elapsed / minutes


def _pace_adjust(df, fraction: Optional[float]):
    """盤中：今天的量只累積了一部分，依開盤經過比例換算成全日估計量，點火量能門檻才有意義。"""
    if fraction is None or df is None or df.empty:
        return df
    df = df.copy()
    df.iloc[-1, df.columns.get_loc("Volume")] = float(df["Volume"].iloc[-1]) / max(fraction, 0.15)
    return df


# ===== 分析 ===================================================================
def _report(symbol: str, fraction: Optional[float], lightweight: bool):
    from src.simple_signal import download_prices, build_report
    period = "2y" if not lightweight else "1y"
    try:
        df = download_prices(symbol, period)
    except Exception:
        df = None
    if (df is None or df.empty) and symbol.endswith(".TW"):
        # 上櫃股/部分 ETF（如債券 ETF）在 Yahoo 是 .TWO
        df = download_prices(symbol[:-3] + ".TWO", period)
    if df is None or df.empty:
        return None
    return build_report(symbol, _pace_adjust(df, fraction), fetch_fundamentals=False, lightweight=lightweight)


def _usd_twd() -> float:
    try:
        import yfinance as yf
        px = yf.Ticker("TWD=X").history(period="5d")["Close"].dropna()
        return float(px.iloc[-1]) if len(px) else 32.0
    except Exception:
        return 32.0


def holding_alerts(holdings: List[Dict], markets: set, fractions: Dict[str, Optional[float]]) -> List[Dict]:
    from src.holding_rules import ADD, CONCENTRATION_LIMIT_PCT, build_holding_verdict
    fx = _usd_twd()
    rows = []
    for h in holdings:
        mkt = "tw" if _is_tw(h["symbol"]) else "us"
        try:
            r = _report(h["symbol"], fractions.get(mkt), lightweight=False)
        except Exception as e:
            print(f"[alerts] 持股分析失敗（{'略' if IN_CI else h['symbol']}）：{type(e).__name__}")
            continue
        if r is None:
            continue
        value_twd = r.latest_close * h["shares"] * (1.0 if mkt == "tw" else fx)
        rows.append({"h": h, "mkt": mkt, "r": r, "value_twd": value_twd})
    total = sum(x["value_twd"] for x in rows) or 1.0

    alerts = []
    for x in rows:
        if x["mkt"] not in markets:
            continue
        h, r = x["h"], x["r"]
        verdict, urgency, ratio, reason = build_holding_verdict(r, h["cost"])
        weight = x["value_twd"] / total * 100
        pnl = (r.latest_close / h["cost"] - 1) * 100 if h["cost"] else None
        if verdict == ADD and weight >= CONCENTRATION_LIMIT_PCT:
            verdict, ratio = "續抱觀察", "0%"
            reason = f"出現點火，但已佔總資產 {weight:.0f}%（上限 {CONCENTRATION_LIMIT_PCT:.0f}%），不再加碼。"
        actionable = verdict in {ADD, "分批減碼", "觀察減碼", "停損出場", "獲利了結"}
        if actionable:
            alerts.append({
                "kind": "holding", "symbol": h["symbol"], "verdict": verdict, "ratio": ratio,
                "price": r.latest_close, "pnl": pnl, "weight": weight, "reason": reason,
                "key": f"holding:{h['symbol']}:{verdict}",
            })
        elif weight > CONCENTRATION_LIMIT_PCT + 5:
            alerts.append({
                "kind": "holding", "symbol": h["symbol"], "verdict": "集中度過高", "ratio": "降至 15%",
                "price": r.latest_close, "pnl": pnl, "weight": weight,
                "reason": f"單一持股佔總資產 {weight:.0f}%，超過 {CONCENTRATION_LIMIT_PCT:.0f}% 上限；逢強分批調節，降低單一股票風險。",
                "key": f"conc:{h['symbol']}",
            })
    return alerts


def watchlist(markets: set, exclude: set) -> List[str]:
    from src.ai_mainline_backtest import AI_MAINLINE_UNIVERSE
    from src.pipeline.daily_report import TW_UNIVERSE, US_UNIVERSE
    syms: List[str] = []
    if "us" in markets:
        syms += AI_MAINLINE_UNIVERSE["us"] + US_UNIVERSE
    if "tw" in markets:
        syms += AI_MAINLINE_UNIVERSE["tw"] + [_yf_symbol(t) for t in TW_UNIVERSE]
    out, seen = [], set(exclude)
    for s in syms:
        s = _yf_symbol(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def buy_alerts(symbols: List[str], fractions: Dict[str, Optional[float]]) -> List[Dict]:
    from src.simple_signal import map_ai_chain_and_bottleneck
    hits = []
    for s in symbols:
        mkt = "tw" if _is_tw(s) else "us"
        try:
            r = _report(s, fractions.get(mkt), lightweight=True)
        except Exception:
            continue
        if r is None or r.ignition_days_ago is None or r.ignition_days_ago > 1:
            continue
        if r.ma120 and r.latest_close < r.ma120 * 0.85:
            continue  # 深度空頭的反彈長紅不追（與 derive_today_plan 一致）
        layer = map_ai_chain_and_bottleneck(s, "")[0]
        hits.append({
            "kind": "buy", "symbol": s, "verdict": "點火買點", "price": r.latest_close,
            "layer": layer, "chg": r.day_change_pct, "vr": r.volume_ratio,
            "entry": r.today_entry_zone, "reason": r.today_note,
            "key": f"buy:{s}",
        })
    hits.sort(key=lambda a: (a["layer"] is not None, a["chg"]), reverse=True)
    return hits[:MAX_BUY_ALERTS]


# ===== 去重狀態 ===============================================================
def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def load_state() -> Dict[str, str]:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    cutoff = (datetime.now(TPE) - timedelta(days=30)).isoformat()
    state = {k: v for k, v in state.items() if v >= cutoff}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def filter_new(alerts: List[Dict], state: Dict[str, str]) -> List[Dict]:
    cutoff = (datetime.now(TPE) - timedelta(days=DEDUP_DAYS)).isoformat()
    return [a for a in alerts if state.get(_hash(a["key"]), "") < cutoff]


# ===== 訊息與推播 =============================================================
def format_message(holding: List[Dict], buys: List[Dict]) -> str:
    now = datetime.now(TPE).strftime("%m/%d %H:%M")
    lines = [f"📣 AI 主線策略提醒 {now}（台北）"]
    if holding:
        lines.append("\n【持股】")
        for a in holding:
            pnl = f"，損益 {a['pnl']:+.1f}%" if a.get("pnl") is not None else ""
            lines.append(f"• {a['symbol']} {a['verdict']} {a['ratio']}｜現價 {a['price']:.2f}{pnl}，佔 {a['weight']:.0f}%")
            lines.append(f"  {a['reason']}")
    if buys:
        lines.append("\n【新買點 · 爆量長紅點火】")
        for a in buys:
            layer = f"［{a['layer']}］" if a.get("layer") else ""
            lines.append(f"• {a['symbol']}{layer} 今日 {a['chg']:+.1f}%｜現價 {a['price']:.2f}｜進場區 {a['entry']}")
            lines.append(f"  {a['reason']}")
    lines.append("\n規則化訊號，非投資建議；下單前請自行確認。")
    return "\n".join(lines)


def send(title: str, body: str) -> List[str]:
    sent = []
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        r = requests.post(f"{server}/{topic}", data=body.encode("utf-8"),
                          headers={"Title": title.encode("utf-8"), "Priority": "high", "Tags": "chart_with_upwards_trend"},
                          timeout=20)
        r.raise_for_status()
        sent.append("ntfy")
    tg_token, tg_chat = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tg_token and tg_chat:
        for i in range(0, len(body), 3800):  # Telegram 單則上限 4096 字
            r = requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                              json={"chat_id": tg_chat, "text": body[i:i + 3800]}, timeout=20)
            r.raise_for_status()
        sent.append("telegram")
    hook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if hook:
        for i in range(0, len(body), 1900):  # Discord 單則上限 2000 字
            r = requests.post(hook, json={"content": body[i:i + 1900]}, timeout=20)
            r.raise_for_status()
        sent.append("discord")
    host, to = os.environ.get("SMTP_HOST", "").strip(), os.environ.get("ALERT_EMAIL_TO", "").strip()
    if host and to:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"], msg["From"], msg["To"] = title, os.environ.get("SMTP_USER", to), to
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=30) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
            s.sendmail(msg["From"], [to], msg.as_string())
        sent.append("email")
    return sent


# ===== 主流程 =================================================================
def resolve_markets(arg: str) -> tuple[set, Dict[str, Optional[float]]]:
    fractions = {"us": session_fraction("us"), "tw": session_fraction("tw")}
    if arg == "auto":
        markets = {m for m, f in fractions.items() if f is not None}
        return markets, fractions
    if arg in ("us", "tw"):
        return {arg}, fractions
    return {"us", "tw"}, {"us": None, "tw": None}   # all：收盤後，不做量能換算


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="all", choices=["auto", "us", "tw", "all"])
    ap.add_argument("--dry-run", action="store_true", help="只印出、不推播、不更新去重狀態")
    args = ap.parse_args(argv)

    markets, fractions = resolve_markets(args.market)
    if not markets:
        print("[alerts] 目前台美股皆未開盤，略過。")
        return 0

    holdings = load_holdings()
    held = {h["symbol"] for h in holdings}
    print(f"[alerts] markets={sorted(markets)} holdings={len(holdings)}")

    h_alerts = holding_alerts(holdings, markets, fractions) if holdings else []
    b_alerts = buy_alerts(watchlist(markets, exclude=held), fractions)

    state = load_state()
    h_new, b_new = filter_new(h_alerts, state), filter_new(b_alerts, state)
    print(f"[alerts] 持股提醒 {len(h_new)}/{len(h_alerts)}、買點提醒 {len(b_new)}/{len(b_alerts)}（新/全部）")
    if not h_new and not b_new:
        return 0

    body = format_message(h_new, b_new)
    title = f"策略提醒：持股 {len(h_new)} 則、買點 {len(b_new)} 則"
    if args.dry_run or not IN_CI:
        print(body)  # 本機才印內容；CI 日誌公開，絕不印出
    if args.dry_run:
        return 0

    channels = send(title, body)
    if not channels:
        print("[alerts] 尚未設定任何推播管道（NTFY_TOPIC / TELEGRAM_* / DISCORD_WEBHOOK_URL / SMTP_*），本次只產生不推送。")
        return 0
    now = datetime.now(TPE).isoformat()
    for a in h_new + b_new:
        state[_hash(a["key"])] = now
    save_state(state)
    print(f"[alerts] 已推播：{', '.join(channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
