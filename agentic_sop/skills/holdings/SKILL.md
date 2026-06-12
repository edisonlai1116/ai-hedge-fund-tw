# Skill: holdings

解析持股成本檔，輸出 `positions@1` artifact。

## 輸入
純文字檔，每行 `代號 平均成本 股數`，例如：

```
AAPL 180 50
2330 980 5
```

`#` 開頭為註解、空行忽略。代號可為美股或台股（含 `00403A` 這類英數代號）。

## 輸出（positions@1）
`data.positions[]`（`ticker / avg_cost / shares / cost_value`）、`position_count`、`total_cost_value`、`skipped[]`，並附 `trace`（每個數字溯回輸入行）。

## 紀律
非空非註解行若無法解析成 3 欄乾淨持股 → 記入 `skipped` 並 **fail-loud（exit 3）**，除非加 `--allow-unparsed`。不靜默丟資料。

## 用法
```
python skills/holdings/tool.py --in ../股票成本.txt --out runs/x/positions.json
```
