param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $projectRoot "app\frontend"
$nodeDir = Join-Path $projectRoot ".local\node-v22.14.0-win-x64"
$nodeExe = Join-Path $nodeDir "node.exe"
$npmCmd = Join-Path $nodeDir "npm.cmd"
$frontendServerScript = Join-Path $frontendDir "scripts\serve-dist.cjs"
$poetryCommand = Get-Command poetry -ErrorAction SilentlyContinue
$poetryExe = if ($poetryCommand) { $poetryCommand.Source } else { Join-Path $env:AppData "Python\Scripts\poetry.exe" }

if (-not (Test-Path $poetryExe)) {
    Write-Host "Poetry is not installed." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $nodeExe) -or -not (Test-Path $npmCmd)) {
    Write-Host "Portable Node.js was not found in .local. Please set up the frontend dependencies first." -ForegroundColor Red
    exit 1
}

$backendCommand = "& '$poetryExe' run uvicorn app.backend.main:app --reload --port $BackendPort"
$frontendCommand = @"
`$env:Path = '$nodeDir;' + `$env:Path;
Write-Host 'Building frontend (dist)...' -ForegroundColor Cyan
& '$npmCmd' run build
& '$nodeExe' '$frontendServerScript' $FrontendPort
"@

Write-Host "Starting backend on http://localhost:$BackendPort" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand -WorkingDirectory $projectRoot

Write-Host "Starting frontend on http://localhost:$FrontendPort" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand -WorkingDirectory $frontendDir

Write-Host "Frontend: http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "Backend:  http://localhost:$BackendPort" -ForegroundColor Green
