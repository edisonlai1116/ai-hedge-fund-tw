param(
    [Parameter(Mandatory = $true)]
    [string[]]$Tickers,

    [string]$StartDate,

    [string]$EndDate,

    [string]$Model = "gpt-4.1",

    [string[]]$Analysts,

    [double]$InitialCash = 100000.0,

    [double]$MarginRequirement = 0.0,

    [switch]$ShowReasoning
)

$ErrorActionPreference = "Stop"

$poetryCommand = Get-Command poetry -ErrorAction SilentlyContinue
$poetryExe = if ($poetryCommand) { $poetryCommand.Source } else { Join-Path $env:AppData "Python\Scripts\poetry.exe" }

if (-not (Test-Path $poetryExe)) {
    Write-Host "Poetry is not installed. Install it first, then rerun this script." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "Missing .env. Copy .env.example to .env and add your API keys first." -ForegroundColor Yellow
    } else {
        Write-Host "Missing both .env and .env.example. Check that the project files are complete." -ForegroundColor Red
    }
    exit 1
}

$tickerArg = $Tickers -join ","
$arguments = @(
    "run",
    "python",
    "src/main.py",
    "--tickers", $tickerArg,
    "--model", $Model,
    "--initial-cash", $InitialCash,
    "--margin-requirement", $MarginRequirement
)

if ($StartDate) {
    $arguments += @("--start-date", $StartDate)
}

if ($EndDate) {
    $arguments += @("--end-date", $EndDate)
}

if ($Analysts -and $Analysts.Count -gt 0) {
    $arguments += @("--analysts", ($Analysts -join ","))
}

if ($ShowReasoning) {
    $arguments += "--show-reasoning"
}

Write-Host "Running AI Hedge Fund analysis for: $tickerArg" -ForegroundColor Cyan
& $poetryExe @arguments
