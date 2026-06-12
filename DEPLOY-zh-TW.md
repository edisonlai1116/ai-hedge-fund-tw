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
