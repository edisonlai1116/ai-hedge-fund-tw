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
BACKTEST_MAX_AGE_DAYS = 7


def _write(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


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

    if args.force_backtest or _backtest_stale():
        from src.strategy.backtest import build_backtest_report
        bt = build_backtest_report()
        if bt["markets"]:
            _write(bt, BACKTEST_JSON)
            for m, d in bt["markets"].items():
                p = d["periods"]["全期"]
                print(f"[strategy_report] 回測 {m}: 年化 {p['strategy']['cagr_pct']}% vs {d['benchmark_symbol']} {p['benchmark']['cagr_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
