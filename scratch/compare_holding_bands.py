"""比較「3–6 個月波段」vs「6–18 個月波段」哪個回測較好。

同一套 AI 主線股、同一期間，只改持有期間／停利／移動停損三個定義波段的參數，
跑多組配置並列出關鍵績效，最後依「年化報酬 + 超額報酬 + Sharpe」綜合給出較佳者。
"""
import sys
from src.ai_mainline_backtest import run_ai_mainline_backtest

MARKET = sys.argv[1] if len(sys.argv) > 1 else "us"
PERIOD = sys.argv[2] if len(sys.argv) > 2 else "5y"

# 波段配置：名稱 -> (停利%, 移動停損%, 最長持有天數)
CONFIGS = {
    "3–6 月波段（短中）": dict(take_profit_pct=20.0, trailing_stop_pct=10.0, max_holding_days=126),
    "6–18 月波段（中長·現行）": dict(take_profit_pct=35.0, trailing_stop_pct=18.0, max_holding_days=378),
    # 控制變因：只改持有上限、停利/停損維持現行，隔離「持有期間」單一效果
    "短持有(僅縮上限126)": dict(take_profit_pct=35.0, trailing_stop_pct=18.0, max_holding_days=126),
}

KEYS = [
    ("total_return_pct", "累積報酬%"),
    ("cagr_pct", "年化CAGR%"),
    ("excess_return_pct", "超額(vs對標)%"),
    ("sharpe_ratio", "Sharpe"),
    ("max_drawdown_pct", "最大回撤%"),
    ("win_rate", "勝率%"),
    ("total_trades", "交易數"),
    ("avg_holding_days", "平均持有天"),
]

results = {}
for name, params in CONFIGS.items():
    print(f"\n>>> 跑 {name} ({MARKET}, {PERIOD}) ...", flush=True)
    try:
        r = run_ai_mainline_backtest(market=MARKET, period=PERIOD, **params)
        results[name] = r
        bench = r.get("benchmark_return_pct")
        print(f"    完成：累積 {r['total_return_pct']:+.1f}% / CAGR {r['cagr_pct']:+.1f}% / "
              f"對標 {bench:+.1f}% / 超額 {r['excess_return_pct']:+.1f}% / Sharpe {r['sharpe_ratio']:.2f} / "
              f"回撤 {r['max_drawdown_pct']:.1f}% / 勝率 {r['win_rate']:.0f}% / "
              f"{r['total_trades']} 筆 / 平均持有 {r['avg_holding_days']:.0f} 天", flush=True)
    except Exception as e:
        print(f"    失敗：{e}", flush=True)

if results:
    print("\n================ 對照表 ================")
    names = list(results.keys())
    header = "指標".ljust(16) + "".join(n[:18].ljust(20) for n in names)
    print(header)
    for key, label in KEYS:
        row = label.ljust(16)
        for n in names:
            v = results[n].get(key)
            row += (f"{v:.2f}" if isinstance(v, float) else str(v)).ljust(20)
        print(row)

    # 綜合評分：年化報酬 + 超額報酬 + Sharpe*5 - 回撤*0.3（回撤為正數，越小越好）
    print("\n================ 綜合評分（越高越好）================")
    best, best_score = None, -1e18
    for n in names:
        r = results[n]
        score = r["cagr_pct"] + r["excess_return_pct"] + r["sharpe_ratio"] * 5 - r["max_drawdown_pct"] * 0.3
        print(f"  {n:24s} 評分 {score:8.2f}  "
              f"(CAGR {r['cagr_pct']:+.1f} + 超額 {r['excess_return_pct']:+.1f} + Sharpe×5 {r['sharpe_ratio']*5:+.1f} - 回撤×0.3 {r['max_drawdown_pct']*0.3:.1f})")
        if score > best_score:
            best, best_score = n, score
    print(f"\n★ 較佳波段：{best}")
