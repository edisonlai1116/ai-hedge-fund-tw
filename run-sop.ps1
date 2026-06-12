# run-sop.ps1 — 以 agentic-sop kit 跑「持股檢視 SOP」（DRAFT，需人覆核）
# 預設讀專案根目錄的 股票成本.txt；--Plan 只列計畫不執行。
param(
    [string]$Flow = "workflow/stock_review.json",
    [string]$InputFile = "../股票成本.txt",
    [string]$RunId = "",
    [switch]$Plan,
    [switch]$Selftest
)

# 注意：不設 $ErrorActionPreference="Stop"，否則 PowerShell 會把 python 寫到 stderr 的
# unittest 進度/警告當成致命錯誤。python 的真正失敗仍會反映在離開碼。
$env:PYTHONUTF8 = "1"   # Windows 中文語系 (cp950) 下避免 emoji/中文輸出編碼錯誤
$kit = Join-Path $PSScriptRoot "agentic_sop"
Push-Location $kit
try {
    if ($Selftest) {
        python selftest.py
        return
    }
    $args = @("workflow/run.py", "--flow", $Flow)
    if ($Plan)        { $args += "--plan" }
    else              { $args += @("--input", $InputFile) }
    if ($RunId -ne "") { $args += @("--run-id", $RunId) }
    python @args
}
finally {
    Pop-Location
}
