"""自動買賣提醒（GitHub Actions 排程呼叫；免 API Key）。規則來源：src.strategy.momentum（單一策略）。

推播時機（條件成立才推，同一訊號不重複）：
  1) 每月調整（每月前 3 個平日）：你的持股該 加碼 / 減碼 / 賣出換股 的清單，以及買進區中你還沒有的「新買進」。
     與回測同規則、同頻率——月中不會因排名小幅變動叫你買賣。
  2) 點火事件（任何交易日、盤中也檢查）：名單內（持股中仍在續抱區、或買進區）的股票出現
     「爆量長紅點火」（單日 ≥+5%、量 ≥1.3 倍均量）→ 提醒可提前加碼／買進。

持股來源：GitHub Secret `HOLDINGS`（每行「代號 成本 股數」），或本機 股票成本.txt；網站「我的持股」頁可一鍵匯出。
推播管道（擇一或多個）：NTFY_TOPIC / TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID / DISCORD_WEBHOOK_URL / SMTP_*。
隱私：repo 公開、Actions 日誌公開——CI 中只印筆數，不印持股與訊息內容。

用法：python -m src.pipeline.alerts [--market auto|us|tw|all] [--dry-run] [--force-rebalance]
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
STRATEGY_JSON = os.path.join(REPO_ROOT, "docs", "data", "strategy.json")
STATE_PATH = os.environ.get("ALERT_STATE_PATH", os.path.join(REPO_ROOT, ".alert_state", "state.json"))
EVENT_DEDUP_DAYS = 5
IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"
TPE = timezone(timedelta(hours=8))


# ===== 持股解析 =============================================================
def parse_holdings(text: str) -> List[Dict]:
    """每行 `代號 成本 股數`，# 開頭為註解。同代號合併（加權平均成本）。"""
    from src.strategy.momentum import to_yf
    merged: Dict[str, Dict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        try:
            cost, shares = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        sym = to_yf(parts[0])
        if sym in merged:
            m = merged[sym]
            total = m["shares"] + shares
            m["cost"] = (m["cost"] * m["shares"] + cost * shares) / total if total else cost
            m["shares"] = total
        else:
            merged[sym] = {"symbol": sym, "cost": cost, "shares": shares}
    return list(merged.values())


def load_holdings() -> List[Dict]:
    text = os.environ.get("HOLDINGS", "")
    if not text.strip() and os.path.exists(HOLDINGS_TXT):
        with open(HOLDINGS_TXT, encoding="utf-8") as f:
            text = f.read()
    return parse_holdings(text)


# ===== 盤中量能換算 =========================================================
def session_fraction(market: str, now_utc: Optional[datetime] = None) -> Optional[float]:
    """該市場今日盤中已經過的比例（0~1）；未開盤或已收盤回 None。"""
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


def live_ignitions(symbols: List[str], fractions: Dict[str, Optional[float]]) -> Dict[str, Dict]:
    """抓最新日線（盤中把今天的量依開盤經過比例換算成全日），回傳有點火的 {symbol: ignition}。"""
    from src.simple_signal import detect_ignition
    from src.strategy.momentum import download_closes, market_of
    out: Dict[str, Dict] = {}
    if not symbols:
        return out
    for sym, f in download_closes(symbols, period="3mo").items():
        frac = fractions.get(market_of(sym))
        if frac is not None and len(f):
            f = f.copy()
            f.iloc[-1, f.columns.get_loc("Volume")] = float(f["Volume"].iloc[-1]) / max(frac, 0.15)
        ign = detect_ignition(f)
        if ign.get("ignition_days_ago") is not None and ign["ignition_days_ago"] <= 1:
            ign["close"] = round(float(f["Close"].iloc[-1]), 2)
            out[sym] = ign
    return out


# ===== 產生提醒 ===============================================================
def build_alerts(holdings: List[Dict], report: Dict, markets: set, fractions: Dict[str, Optional[float]],
                 rebalance: bool, fx: float) -> List[Dict]:
    from src.strategy.momentum import TOP_N, evaluate_holdings
    ev = evaluate_holdings(holdings, report, fx) if holdings else {"holdings": [], "new_buys": {"us": [], "tw": []}}
    month = datetime.now(TPE).strftime("%Y-%m")
    alerts: List[Dict] = []

    if rebalance:
        for h in ev["holdings"]:
            if h["market"] in markets and h["action"] in ("賣出換股", "減碼", "加碼"):
                alerts.append({"group": "rebalance", "symbol": h["symbol"], "action": h["action"],
                               "price": h["price"], "pnl": h["pnl_pct"], "weight": h["weight_pct"],
                               "reason": h["reason"], "key": f"reb:{month}:{h['symbol']}:{h['action']}"})
        for m in markets:
            for b in ev["new_buys"].get(m, []):
                tgt = f"，目標約 NT${b['target_twd']:,.0f}" if b.get("target_twd") else ""
                alerts.append({"group": "rebalance", "symbol": b["symbol"], "action": "新買進",
                               "price": b["close"], "pnl": None, "weight": None,
                               "reason": f"動能排名第 {b['rank']} 名（近 6 個月 {b['ret_6m_pct']:+.0f}%）{tgt}。",
                               "key": f"reb:{month}:{b['symbol']}:new"})

    # 點火事件：持股中仍在名單內者 + 前 10 名
    watch = {h["symbol"]: "持股" for h in ev["holdings"]
             if h["action"] in ("續抱", "加碼") and h["market"] in markets}
    for m in markets:
        for r in report.get("markets", {}).get(m, {}).get("rows", [])[:TOP_N[m]]:
            watch.setdefault(r["symbol"], f"前 {TOP_N[m]} 名")
    for sym, ign in live_ignitions(list(watch), fractions).items():
        when = "今天" if ign["ignition_days_ago"] == 0 else "昨天"
        what = "可加碼" if watch[sym] == "持股" else "可提前買進"
        alerts.append({"group": "event", "symbol": sym, "action": f"點火·{what}", "price": ign["close"],
                       "pnl": None, "weight": None,
                       "reason": (f"{watch[sym]}標的{when}爆量長紅 +{ign['ignition_gain_pct']}%（量 {ign['ignition_volume_ratio']} 倍）；"
                                  f"停損參考點火日低點 {ign['ignition_low']}。"),
                       "key": f"ign:{sym}"})
    return alerts


# ===== 去重 ===================================================================
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
    cutoff = (datetime.now(TPE) - timedelta(days=45)).isoformat()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in state.items() if v >= cutoff}, f)


def filter_new(alerts: List[Dict], state: Dict[str, str]) -> List[Dict]:
    """月調提醒的 key 含月份（每月一次）；點火事件 5 天內不重複。"""
    cutoff = (datetime.now(TPE) - timedelta(days=EVENT_DEDUP_DAYS)).isoformat()
    out = []
    for a in alerts:
        seen = state.get(_hash(a["key"]), "")
        if seen and (a["key"].startswith("reb:") or seen >= cutoff):
            continue
        out.append(a)
    return out


# ===== 訊息與推播 =============================================================
def format_message(alerts: List[Dict], report: Dict) -> str:
    now = datetime.now(TPE).strftime("%m/%d %H:%M")
    lines = [f"📣 Sharpe 動能策略提醒 {now}（台北）"]
    reb = [a for a in alerts if a["group"] == "rebalance"]
    ev = [a for a in alerts if a["group"] == "event"]
    if reb:
        lines.append("\n【每月調整】")
        for label in ("賣出換股", "減碼", "加碼", "新買進"):
            for a in [x for x in reb if x["action"] == label]:
                pnl = f"，損益 {a['pnl']:+.1f}%" if a.get("pnl") is not None else ""
                wt = f"，佔 {a['weight']:.0f}%" if a.get("weight") is not None else ""
                lines.append(f"• {label} {a['symbol']}｜現價 {a['price']}{pnl}{wt}")
                lines.append(f"  {a['reason']}")
    if ev:
        lines.append("\n【點火事件】")
        for a in ev:
            lines.append(f"• {a['action']} {a['symbol']}｜現價 {a['price']}")
            lines.append(f"  {a['reason']}")
    nxt = report.get("strategy", {}).get("next_rebalance")
    if nxt:
        lines.append(f"\n下次月調：{nxt}。規則化訊號，非投資建議；下單前請自行確認。")
    return "\n".join(lines)


def send(title: str, body: str) -> List[str]:
    sent = []
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        server = os.environ.get("NTFY_SERVER", "").strip() or "https://ntfy.sh"
        r = requests.post(f"{server.rstrip('/')}/{topic}", data=body.encode("utf-8"),
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
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "") or 587), timeout=30) as s:
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
        return {m for m, f in fractions.items() if f is not None}, fractions
    if arg in ("us", "tw"):
        return {arg}, fractions
    return {"us", "tw"}, {"us": None, "tw": None}   # all：收盤後，不做量能換算


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="all", choices=["auto", "us", "tw", "all"])
    ap.add_argument("--dry-run", action="store_true", help="只印出、不推播、不更新去重狀態")
    ap.add_argument("--force-rebalance", action="store_true", help="不論日期都產生月調清單（測試用）")
    args = ap.parse_args(argv)

    markets, fractions = resolve_markets(args.market)
    if not markets:
        print("[alerts] 目前台美股皆未開盤，略過。")
        return 0
    try:
        with open(STRATEGY_JSON, encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        print(f"[alerts] 讀不到策略排名 {STRATEGY_JSON}：{e}")
        return 1

    from src.strategy.momentum import is_rebalance_window, usd_twd
    rebalance = args.force_rebalance or (args.market == "all" and is_rebalance_window())
    holdings = load_holdings()
    print(f"[alerts] markets={sorted(markets)} holdings={len(holdings)} rebalance={rebalance}")

    alerts = build_alerts(holdings, report, markets, fractions, rebalance, usd_twd())
    state = load_state()
    new = filter_new(alerts, state)
    print(f"[alerts] 提醒 {len(new)}/{len(alerts)}（新/全部）")
    if not new:
        return 0

    body = format_message(new, report)
    n_reb = sum(1 for a in new if a["group"] == "rebalance")
    title = f"策略提醒：月調 {n_reb} 則、點火 {len(new) - n_reb} 則"
    if args.dry_run or not IN_CI:
        print(body)   # 本機才印內容；CI 日誌公開，絕不印出
    if args.dry_run:
        return 0

    channels = send(title, body)
    if not channels:
        print("[alerts] 尚未設定任何推播管道（NTFY_TOPIC / TELEGRAM_* / DISCORD_WEBHOOK_URL / SMTP_*），本次只產生不推送。")
        return 0
    now = datetime.now(TPE).isoformat()
    for a in new:
        state[_hash(a["key"])] = now
    save_state(state)
    print(f"[alerts] 已推播：{', '.join(channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
