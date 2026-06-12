# agentic-sop kit — 整合說明（本專案）

本資料夾是 [`agentic-sop-to-work`](https://github.com/s0912758806p/agentic-sop-to-work) 的可攜 kit，
複製進 `ai-hedge-fund`，作為**後續優化系統的安全方法論層**。它讓每次調整都走可重複、可驗證、不臆造、
產出一律標 DRAFT 並需人覆核的流程。

## 為什麼放這套
LLM 直接做投資決策有兩個風險：**捏造**（編造不存在的數字）與**不可重現**。這套 kit 用三個機制壓住：
- **邏輯在程式、不在模型**：每個 skill 是純 Python 工具，讀寫固定 schema 的 JSON artifact。
- **逐步 gate 驗證**：`schema_gate`（必要欄位）、`recompute_gate`（重算核對，如持股筆數）、
  `trace_gate`（每個數字須溯回輸入，防捏造）、`cmd_gate`。
- **缺資料標 `【待補】`、產出標 DRAFT**：永不自動歸檔進任何受控系統。

## Windows 注意事項（重要）
本機中文語系預設 cp950，kit 會輸出 emoji/中文，須用 **UTF-8 模式**跑，否則 console 編碼會報錯：

```powershell
$env:PYTHONUTF8 = "1"      # 或每次：python -X utf8 ...
```

最簡單是用專案根目錄的 `run-sop.ps1`（已自動設好）。

## 快速開始
```powershell
# 自我測試（5 項，全綠才算 kit 健康）
.\run-sop.ps1 -Selftest

# 只看計畫、不執行
.\run-sop.ps1 -Plan

# 用你真實的 股票成本.txt 跑持股檢視 SOP（產出 DRAFT）
.\run-sop.ps1

# 指定輸入與 run-id
.\run-sop.ps1 -Input ..\股票成本.txt -RunId today
```
產出在 `agentic_sop/runs/<run-id>/position_review.md`。

## 內建的股票檢視 SOP（`workflow/stock_review.json`）
1. `holdings`：解析 `代號 成本 股數` → positions artifact（附來源追溯）。
2. `verify_count`：`recompute_gate` 重算持股筆數，核對 `position_count`。
3. `position_review`：輸出 DRAFT 持股表；**現價／未實現損益／股癌共識**一律標 `【待補】`。

## 如何擴充（把股癌等 live 資料接進來）
`position_review` 刻意把需要即時資料的欄位留成 `【待補】`，由 live 系統填入後人工覆核：
- 股癌／輿情共識 → `src/sentiment/consensus_engine.py`（`WeightedConsensusEngine`）與
  `src/agents/gooaye_sentiment.py`。
- 行情／現價 → `src/tools/api.py` 或免 Key 的 `src/simple_signal.py`。

新增 skill 用範本：`python new_skill.py <name>`（會在 `skills/<name>/` 生成 tool.py 與 SKILL.md），
再把該步驟加進 flow.json，並視需要掛上對應 gate。完整方法論見 `SOP.md`。
