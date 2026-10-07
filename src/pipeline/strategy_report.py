"""每日產生策略排名（docs/data/strategy.json）；回測結果每 7 天重算一次（docs/data/strategy_backtest.json）。

用法：python -m src.pipeline.strategy_report [--force-backtest]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
STRATEGY_JSON = os.path.join(DATA_DIR, "strategy.json")
BACKTEST_JSON = os.path.join(DATA_DIR, "strategy_backtest.json")
OPPORTUNITY_JSON = os.path.join(DATA_DIR, "opportunity.json")
SIGNAL_LOG = os.path.join(DATA_DIR, "signal_log.jsonl")
SIGNAL_ACCURACY_JSON = os.path.join(DATA_DIR, "signal_accuracy.json")
ADVICE_JSON = os.path.join(DATA_DIR, "daily_advice.json")
ADVICE_LOG = os.path.join(DATA_DIR, "advice_log.jsonl")
BACKTEST_MAX_AGE_DAYS = 7


def _write(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def drawdown_map(opp: dict) -> dict:
    """{代號: {type, why, status}}：機會評分的回撤分類，給每日建議與持股建議共用（同一套規則）。"""
    return {r["ticker"]: {"type": r.get("drawdown_type"), "why": (r.get("drawdown") or {}).get("why"), "status": r.get("status"),
                          "confirmations": r.get("confirmations") or []}
            for r in opp.get("ranking", []) if r.get("drawdown_type")}


def _backtest_stale() -> bool:
    try:
        with open(BACKTEST_JSON, encoding="utf-8") as f:
            ts = datetime.fromisoformat(json.load(f)["generated_at"])
        return datetime.now(timezone.utc) - ts > timedelta(days=BACKTEST_MAX_AGE_DAYS)
    except Exception:
        return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-backtest", action="store_true")
    args = ap.parse_args(argv)

    from src.strategy.momentum import build_strategy_report
    report = build_strategy_report()
    if not report["markets"]:
        print("[strategy_report] 兩個市場都排名失敗，保留舊檔不覆蓋。")
        return 1
    _write(report, STRATEGY_JSON)
    for m, d in report["markets"].items():
        top = ", ".join(r["symbol"] for r in d["rows"][:10])
        print(f"[strategy_report] {m}: {d['universe_size']} 檔；前 10：{top}")

    # 機會評分卡（cross-sectional ranking）：驗收清單 + 各市場低檔布局候選 + 動能前段；不含任何持股資訊。
    try:
        from src.ranking.engine import rank_stocks
        from src.ranking.themes import ACCEPTANCE_TICKERS
        from src.ranking import tracking
        watch = list(ACCEPTANCE_TICKERS)
        for m in ("us", "tw"):
            watch += report["markets"].get(m, {}).get("low_entry", [])[:15]         # 低檔布局：全部做回撤分類（大跌 ≠ 便宜）
            watch += [r["symbol"] for r in report["markets"].get(m, {}).get("rows", [])[:6]]   # 動能前段
        watch = list(dict.fromkeys(watch))
        ranks = {r["symbol"]: (r["rank"], d["universe_size"]) for d in report["markets"].values() for r in d.get("rows", [])}
        opp = rank_stocks(watch, universe_ranks=ranks)
        for r in opp["ranking"]:   # 精簡：原始財報細節不輸出
            for k in ("fundamentals",):
                r.pop(k, None)
        _write(opp, OPPORTUNITY_JSON)
        report["drawdown_types"] = drawdown_map(opp)
        from src.strategy.momentum import low_view
        for d in report["markets"].values():          # 低檔股：價格位置 vs 投資建議（同一函式）
            for r in d.get("rows", []):
                if r.get("low_entry"):
                    r["low_view"] = low_view(r, report, d.get("universe_size"))
        _write(report, STRATEGY_JSON)
        n = tracking.log_signals(opp, SIGNAL_LOG)
        _write(tracking.evaluate(SIGNAL_LOG), SIGNAL_ACCURACY_JSON)
        print(f"[strategy_report] 機會評分 {len(opp['ranking'])} 檔（regime {opp['regime']['regime']}），新增訊號紀錄 {n} 筆")
    except Exception as exc:
        print(f"[strategy_report] 機會評分失敗（不影響策略排名）：{type(exc).__name__}: {exc}")

    if args.force_backtest or _backtest_stale():
        from src.strategy.backtest import build_backtest_report
        bt = build_backtest_report()
        if bt["markets"]:
            _write(bt, BACKTEST_JSON)
            for m, d in bt["markets"].items():
                p = d["periods"]["全期"]
                print(f"[strategy_report] 回測 {m}: 年化 {p['strategy']['cagr_pct']}% vs {d['benchmark_symbol']} {p['benchmark']['cagr_pct']}%")

    # 每日建議（市場情緒 + 低檔/月調/點火）：每天都給結論，多數日子是「今天不用動」。
    try:
        from src.strategy.daily_advice import append_log, build_daily_advice
        bt_json = None
        if os.path.exists(BACKTEST_JSON):
            with open(BACKTEST_JSON, encoding="utf-8") as f:
                bt_json = json.load(f)
        advice = build_daily_advice(report, bt_json)
        _write(advice, ADVICE_JSON)
        append_log(advice, ADVICE_LOG)
        print(f"[strategy_report] 每日建議：{advice['headline']}")
    except Exception as exc:
        print(f"[strategy_report] 每日建議失敗：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
