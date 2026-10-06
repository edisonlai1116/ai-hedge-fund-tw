# 部署清單：推上 GitHub → Render 上線（一頁照做）

目標：把完整網站（查個股四分頁 + 每日 Top 50 + 股癌/輿情）放到網路上，開網址就能用。

---

## 步驟 1：建 GitHub repo 並推上去

1. 開 https://github.com/new
2. Repository name 填：`ai-hedge-fund-tw`
3. 選 **Public（公開）**　←　免費 Pages/部署需要
4. **不要**勾任何 Add README / .gitignore / license
5. 按 **Create repository**
6. 在你電腦的 PowerShell 貼這兩行（第一次會跳瀏覽器登入 GitHub，照按授權）：

```powershell
cd C:\Users\User\Desktop\codex\ai-hedge-fund-main
git push -u origin main
```

確認：開 `https://github.com/edisonlai1116/ai-hedge-fund-tw`，看得到 `app`、`src`、`docs` 等資料夾就成功。

---

## 步驟 2：Render 部署完整網站（查個股四分頁）

1. 開 https://render.com → 用 GitHub 登入
2. 右上 **New +** → **Blueprint**
3. 選 `ai-hedge-fund-tw` repo → **Connect** → 它會讀 `render.yaml` → **Apply / Create**
4. 等第一次建置（約 5–10 分鐘）
5. 完成後點服務名稱，上方會有網址，例如：`https://ai-hedge-fund-tw.onrender.com`

打開那個網址就能用：
- 首頁：個股分析 / 每日掃描 / 持股健檢 / AI 主線回測（你原本的四分頁）
- 網址後面加 `/daily/`：每日台美股 Top 50（秒開版）

> 免費機閒置約 15 分鐘會休眠，休眠後第一次開要等 30–60 秒喚醒，正常。

---

## 步驟 3（選用）：GitHub Pages 放「每日 Top 50」靜態頁

只想要每天自動更新的清單、不想等 Render 喚醒時：
1. repo → **Settings → Pages**
2. Source 選 **Deploy from a branch**，分支 `main`、資料夾 `/docs` → **Save**
3. 開 `https://edisonlai1116.github.io/ai-hedge-fund-tw/`

---

## 之後完全自動
- GitHub Actions 每天 06:30 / 17:00（台北）自動重算 Top 50 並更新。
- 你不用再開 CMD；要查個股就開 Render 網址首頁輸入代號。

## 卡關時
把畫面上的紅字訊息貼給我，我幫你看下一步。

## 自動買賣提醒（GitHub Actions 推播）

`src/pipeline/alerts.py` 會在台股盤中（10:30、13:00）、美股盤中（約開盤後 1 小時／午盤／收盤前）
與每日報告產生後檢查：

規則來源是 `src/strategy/momentum.py`（單一策略「Sharpe 動能輪動」，與網站「我的持股」同一套）：

- **每月調整**（每月前 3 個平日）：持股 加碼／減碼／賣出換股，以及前 10 名中尚未持有的「新買進」。
- **點火事件**（盤中也檢查）：持股中仍在名單內、或前 10 名的股票出現爆量長紅（單日 ≥+5%、量 ≥1.3 倍均量）。

月調提醒每月一次、點火事件 5 天內不重複。設定（repo → Settings → Secrets and variables → Actions → New repository secret）：

| Secret | 說明 |
|---|---|
| `HOLDINGS` | 持股，每行 `代號 成本 股數`（同 股票成本.txt 格式；台股直接寫 2330） |
| `NTFY_TOPIC` | 最簡單：手機裝 ntfy App，訂閱一個難猜的主題名（例：`edison-alerts-8f3k2q`），這裡填同名 |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | 或用 Telegram 機器人（較私密） |
| `DISCORD_WEBHOOK_URL` | 或 Discord 頻道 Webhook |
| `SMTP_HOST` `SMTP_PORT` `SMTP_USER` `SMTP_PASS` `ALERT_EMAIL_TO` | 或 Email（Gmail 需用應用程式密碼） |

repo 是公開的：持股只放 Secret，程式在 Actions 日誌中不印出任何持股內容。
設好後可到 Actions → Trade Alerts → Run workflow（market 選 all）立即測試一次。
本機試跑（只印不推）：`python -m src.pipeline.alerts --market all --dry-run`
