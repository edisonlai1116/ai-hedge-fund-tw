param(
    [Parameter(Mandatory = $true)]
    [string]$Ticker,

    [string]$Market,

    [string]$Period = "6mo",

    # 啟用本地 Ollama AI 委員會（需先安裝 Ollama 並下載模型）
    [switch]$UseAiCommittee,
    [string]$CommitteeModel = "gemma4:e4b"
)

$ErrorActionPreference = "Stop"

$poetryCommand = Get-Command poetry -ErrorAction SilentlyContinue
$poetryExe = if ($poetryCommand) { $poetryCommand.Source } else { Join-Path $env:AppData "Python\Scripts\poetry.exe" }

if (-not (Test-Path $poetryExe)) {
    Write-Host "Poetry is not installed. Install it first, then rerun this script." -ForegroundColor Red
    exit 1
}

$arguments = @(
    "run",
    "python",
    "src/simple_signal.py",
    "--ticker", $Ticker,
    "--period", $Period
)

if ($Market) {
    $arguments += @("--market", $Market)
}

if ($UseAiCommittee) {
    $arguments += "--use-ai-committee"
    $arguments += @("--committee-model", $CommitteeModel)
    Write-Host "AI 委員會：開啟（本地 Ollama 模型 $CommitteeModel）" -ForegroundColor Yellow
}

Write-Host "Running simple signal scan for: $Ticker" -ForegroundColor Cyan
& $poetryExe @arguments
