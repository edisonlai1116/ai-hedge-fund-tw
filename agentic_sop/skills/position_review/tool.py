"""Skill：position_review — 讀 positions artifact，輸出 DRAFT Markdown 持股檢視表。

紀律（對應 SOP 反捏造原則）：
- 只用上游 artifact 內**確實存在**的數字（成本、股數、成本市值、權重）。
- 凡需要外部即時資料的欄位——**現價、未實現損益、股癌/輿情共識分數**——一律標 `【待補】`，
  絕不臆造。這些欄位由 live 系統（src/agents/gooaye_sentiment.py 等）填入後人工覆核。
單一工具（python3 stdlib）；輸出 .md，標 DRAFT、需人覆核。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import kit  # noqa: E402

DEPS = [{"kind": "python", "min": "3.8"}]
WHO = "position_review"
TODO = "【待補】"


def run(inp, out):
    up = kit.read_artifact(inp)
    d = up["data"]
    positions = d.get("positions", [])
    total = d.get("total_cost_value", 0) or 0

    lines = [
        "# 持股檢視 — DRAFT", "",
        "> DRAFT — 需人員覆核；由 agentic-sop-kit 流程產生，非投資建議、非正式紀錄。",
        f"> 持股檔數：{d.get('position_count', len(positions))}　總成本市值：{total}", "",
        "## 持股明細",
        "| 代號 | 平均成本 | 股數 | 成本市值 | 成本權重 | 現價 | 未實現損益 | 股癌/輿情共識 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in positions:
        cv = p.get("cost_value", 0)
        weight = f"{round(cv / total * 100, 1)}%" if total else TODO
        lines.append(
            f"| {p.get('ticker')} | {p.get('avg_cost')} | {p.get('shares')} | {cv} | {weight}"
            f" | {TODO} | {TODO} | {TODO} |"
        )

    skipped = d.get("skipped", [])
    if skipped:
        lines += ["", f"## ⚠️ 未解析行（{len(skipped)} 行，已跳過，需人工確認）"]
        for x in skipped:
            lines.append(f"- line {x.get('line')}: {x.get('text')}")

    lines += [
        "", "## 待補資料（不臆造，交由 live 系統填入後人工覆核）",
        f"- 現價、未實現損益：{TODO}　← 由行情來源（Yahoo Finance / FinancialDatasets）填入",
        f"- 股癌/輿情共識分數：{TODO}　← 由 WeightedConsensusEngine / gooaye_sentiment agent 填入",
        "",
        "## 來源追溯（每個數字溯回輸入位置）",
    ]
    for t in up.get("trace", []):
        lines.append(f"- {t.get('value')} @ {t.get('source')}:{t.get('locator')}")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    kit.skill_main(DEPS, WHO, run)
