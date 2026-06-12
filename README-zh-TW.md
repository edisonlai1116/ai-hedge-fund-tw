# AI Hedge Fund 股票買賣點分析

這份專案是基於 [`virattt/ai-hedge-fund`](https://github.com/virattt/ai-hedge-fund) 建立，目標是讓你可以用 AI analyst workflow 來做股票的買進、賣出與回測分析。

## 你現在可以直接用的內容

- 原始 repo 已下載到這個資料夾
- `run-analysis.ps1`：跑單次股票分析
- `run-backtest.ps1`：跑區間回測
- `run-simple-signal.ps1`：免 API key 的簡化版買賣點分析
- `run-simple-web.ps1`：啟動免 Key 的前端介面

## 目前環境狀態

這台電腦目前還沒有：

- Python
- Poetry
- Git
- Node.js

所以我已經把專案和啟動腳本準備好，但還不能直接執行。

## 最少需要安裝什麼

如果你只想做 CLI 股票分析，不需要 Web UI，最少安裝：

1. Python 3.11
2. Poetry

如果你還想跑圖形化 Web 介面，再加上：

3. Node.js

## API Keys

專案至少需要一組 LLM API key，例如：

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `DEEPSEEK_API_KEY`

如果你分析的股票不只 repo 測試常見的幾檔，通常也需要：

- `FINANCIAL_DATASETS_API_KEY`

建立 `.env` 的方式：

```powershell
Copy-Item .env.example .env
```

然後把 `.env` 裡的 key 換成你自己的值。

## 安裝後的第一次設定

在專案根目錄執行：

```powershell
poetry install
```

## 單次分析

```powershell
.\run-analysis.ps1 -Tickers AAPL,MSFT -StartDate 2025-01-01 -EndDate 2025-03-31 -Model gpt-4.1 -ShowReasoning
```

如果你只想分析單一股票：

```powershell
.\run-analysis.ps1 -Tickers TSLA
```

## 回測

```powershell
.\run-backtest.ps1 -Tickers NVDA -StartDate 2024-01-01 -EndDate 2024-06-30 -Model gpt-4.1
```

## 常用 analyst 名稱

你也可以限制只用部分 analyst：

```powershell
.\run-analysis.ps1 -Tickers AAPL -Analysts warren_buffett,ben_graham,michael_burry
```

常見可用值包含：

- `warren_buffett`
- `ben_graham`
- `michael_burry`
- `peter_lynch`
- `technicals`
- `fundamentals`
- `sentiment`
- `valuation`

## Web 介面

如果之後你想用瀏覽器操作：

```powershell
cd app
.\run.bat
```

但這條路徑還需要 `Node.js`。

## 注意

這個 repo 作者明確標註為教學／研究用途，不是正式投資建議。比較適合拿來做：

- 想法生成
- 分析輔助
- 回測驗證
- 不同 analyst 觀點比較

## 免 Key 簡化版

如果你沒有任何 API key，可以直接用這個版本：

```powershell
.\run-simple-signal.ps1 -Ticker AAPL
```

台股可以直接輸入代碼：

```powershell
.\run-simple-signal.ps1 -Ticker 2330
```

也可以明確指定市場：

```powershell
.\run-simple-signal.ps1 -Ticker 2330 -Market tw
.\run-simple-signal.ps1 -Ticker MSFT -Market us
```

它會輸出：

- 趨勢判斷
- 建議買點區間
- 建議賣點區間
- 建議停損區間

這個簡化版使用 Yahoo Finance 公開行情資料與技術指標，完全不需要 `.env`。

## 免 Key 前端介面

我另外做了一個可直接操作的前端頁面。

啟動方式：

```powershell
cd C:\Users\User\Desktop\codex\ai-hedge-fund-main
.\run-simple-web.ps1
```

啟動後可開啟：

- `http://localhost:5173`

你可以直接輸入：

- 美股：`AAPL`、`MSFT`、`NVDA`
- 台股：`2330`、`2317`

頁面會顯示：

- 趨勢判斷
- 最新收盤
- 買點區間
- 賣點區間
- 停損區間
- 支撐與壓力

## 股癌 (Gooaye) 輿情共識，已列入買賣參考因素

新增分析師 **`gooaye_sentiment`**（顯示名稱：Gooaye (股癌) Consensus），把股癌 Podcast 與社群輿情
納入買賣決策：

- 來源：`src/sentiment/`（`AudioFeedAdapter` 抓 SoundOn RSS、`CustomIngestService` 接 YouTube/網頁、
  `WeightedConsensusEngine` 加權），存進 `app/backend/hedge_fund.db`，後端 API 在 `/sentiment/*`。
- agent：`src/agents/gooaye_sentiment.py` 讀加權共識分數（0–100）→ 訊號（>=65 bullish、<45 bearish、
  其餘 neutral）、confidence = |score-50|×2。已註冊進 `ANALYST_CONFIG`，`portfolio_manager` 會一併納入。
- 穩健性：惰性匯入 + 優雅降級——引擎/DB 不可用或該檔無輿情（多數美股）時回中性、confidence 0，不臆造、不崩潰。

只用股癌分析師跑（台股覆蓋為主）：

```powershell
.\run-analysis.ps1 -Tickers 2330 -Analysts gooaye_sentiment
```

先匯入最新一集股癌（mock fallback 預設開啟，無 ffmpeg 也能跑）：對後端 `POST /sentiment/podcast-scan`。
測試：`tests/test_gooaye_sentiment.py`。

## agentic-sop kit：後續優化的安全方法論層

`agentic_sop/` 是 [`agentic-sop-to-work`](https://github.com/s0912758806p/agentic-sop-to-work) 的可攜 kit，
讓每次系統調整都走「邏輯在程式、逐步 gate 驗證、缺資料標 `【待補】`、產出一律 DRAFT 需人覆核」的流程。

```powershell
.\run-sop.ps1 -Selftest     # kit 自我測試（須 UTF-8 模式，腳本已自動設定）
.\run-sop.ps1 -Plan         # 只看計畫
.\run-sop.ps1               # 用你的 股票成本.txt 產出 DRAFT 持股檢視表
```

詳見 `agentic_sop/INTEGRATION-zh-TW.md` 與 `agentic_sop/SOP.md`。Windows 中文語系須以 UTF-8 模式執行
（`$env:PYTHONUTF8=1`，`run-sop.ps1` 已內建）。

## 每日自動更新 + 開網頁就能看（免本地 CMD）

部署到 GitHub 後，系統會**每天自動產生「每日最該買」報告**，你只要開網頁看，不用再開 CMD。

架構（完全免伺服器、免費、免 Key）：
- **GitHub Actions 排程**（`.github/workflows/daily.yml`，每日 06:30 台北）：跑 `src/pipeline/daily_report.py`，
  掃股癌 RSS 偵測新集數→更新點名、抓總經/地緣新聞輿情、用 Yahoo 技術面對「**美股當日熱門榜 ∪ 股癌點名**」
  （反映每日市場變化；你的持股只用來標記、不主導排序）每檔算綜合買進分數並排序，commit 進 `docs/data/daily_report.json`。
  當日熱門榜來源：`src/sentiment/market_movers.py`（Yahoo 免 Key 篩選 day_gainers / most_actives）。
- **GitHub Pages 靜態網頁**（`docs/index.html`）：讀該 JSON 顯示市場輿情、股癌最新集數、每日最該買排序與
  買賣區間。手機也能看。

綜合買進分數 = 技術面 50% + 股癌共識 25% + 個股新聞 10% + 市場輿情 15%（權重見報告 JSON）。

### 部署步驟（一次性）
1. 先把 repo 推到 GitHub（見最上方）。
2. GitHub repo → **Settings → Pages → Source 選 `Deploy from a branch`，分支 `main`、資料夾 `/docs`**。
3. GitHub repo → **Actions** 分頁 → 開啟 workflow → 可先按 **Run workflow** 手動跑一次。
4. 開 `https://edisonlai1116.github.io/ai-hedge-fund-tw/` 即看每日報告。

> ⚠️ 隱私：GitHub Free 方案下 private repo 的 Pages 站台是**公開**的（有網址就看得到）。
> 因此發布的 JSON **只含代號與分析結果，不含你的成本與股數**（`股票成本.txt` 本身不入版控）。

### 本地手動跑一次（選用，需先 `poetry install`）
```powershell
$env:PYTHONUTF8=1; python -m src.pipeline.daily_report
```

### 新增的輿情 agent
除了股癌，另有 **`macro_news_sentiment`**（總經/地緣輿情）：追蹤川普關稅、戰爭衝突、Fed 利率等
影響大盤的新聞（免 Key 關鍵字情緒），產生市場級多空偏向，已註冊進 `ANALYST_CONFIG`，在完整多師
分析時也會納入。新聞模組為 `src/sentiment/news_feed.py`。

## 把「查個股」互動網頁部署到雲端（Render，免 Key、免費）

GitHub Pages 是靜態網頁，只能顯示每日預算好的清單，**不能讓你在網頁上隨打一檔就查**。要把你原本
「輸入台股/美股個股 → 進出點、買賣建議、規則型 agent 意見」的互動功能放到網路上、不用開 CMD，
就把它部署到一個會「一直開著」的免費後端主機。

已備好的檔案：
- `app/backend/web_app.py`：精簡免-Key 後端（只含 `simple-signals` / `sentiment` 路由 + 服務前端，
  **不含 langchain 大師委員會**，所以小又快），並內建每 2 小時自動掃股癌 RSS 寫入 DB。
- `Dockerfile.web` + `Dockerfile.web.dockerignore`：多階段建置（Node 建前端 → Python 跑後端）。
- `render.yaml`：Render 藍圖，連到 repo 後一鍵建立服務。

### 部署步驟（Render）
1. 先把 repo 推上 GitHub（公開）。
2. 到 https://render.com 用 GitHub 登入。
3. 右上 **New +** → **Blueprint** → 選 `ai-hedge-fund-tw` repo → Render 會讀 `render.yaml` → **Apply**。
4. 等第一次建置（約 5–10 分鐘）。完成後會給你一個網址，例如 `https://ai-hedge-fund-tw.onrender.com`。
5. 開那個網址 → 輸入 `2330` 或 `NVDA` 就能查進出點與買賣建議。

> 免費方案閒置約 15 分鐘會休眠，休眠後第一次開要等約 30–60 秒喚醒，屬正常。
> 完整大師 LLM 委員會（巴菲特等）需要 API key，不在此免-Key 服務內；要用請在本機跑
> `run-analysis.ps1`（見上面「常用 analyst 名稱」）。

### 本機先驗證（選用，需先 `poetry install` 與 build 前端）
```powershell
# 後端（精簡免 Key）
$env:PYTHONUTF8=1; poetry run uvicorn app.backend.web_app:app --port 8000
```
（或直接用既有的 `run-simple-web.ps1`，功能相同。）

### 兩個網址各司其職
- **Render**（互動）：隨打一檔台股/美股 → 進出點、買賣、規則型 agent 意見、股癌輿情。
- **GitHub Pages**（每日總覽）：每天自動算好的「最該買」排序與歷史走勢。
