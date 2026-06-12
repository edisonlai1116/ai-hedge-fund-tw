# Skill: position_review

讀 `positions@1` artifact，輸出 DRAFT Markdown 持股檢視表。

## 輸出
- 持股明細表：代號 / 平均成本 / 股數 / 成本市值 / 成本權重（由上游數字計算）。
- **現價、未實現損益、股癌/輿情共識**：一律標 `【待補】`，由 live 系統填入後人工覆核——不臆造。
- 未解析行警示、來源追溯區、DRAFT 標記。

## 紀律
只輸出上游 artifact 內確實存在的數字；任何需要外部即時資料的欄位都以 `【待補】` 表示缺口，符合 SOP 反捏造原則。

## 用法
```
python skills/position_review/tool.py --in runs/x/positions.json --out runs/x/review.md
```
