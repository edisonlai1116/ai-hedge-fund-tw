param(
    [Parameter(Mandatory = $true)]
    [string]$Ticker,

    [string]$Market,

    [string]$Period = "6mo"
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

Write-Host "Running simple signal scan for: $Ticker" -ForegroundColor Cyan
& $poetryExe @arguments
