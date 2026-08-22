$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run.py"
$config = Join-Path $PSScriptRoot "configs\budget128.json"
$runs = Join-Path $root "runs"
$stdout = Join-Path $runs "paper-coding-128-console.log"
$stderr = Join-Path $runs "paper-coding-128-error.log"
$processFile = Join-Path $runs "paper-coding-128-process.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "OpenRQGM virtual environment is missing: $python"
}
docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon is not ready."
}

New-Item -ItemType Directory -Force -Path $runs | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @($runner, "--config", $config) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    config = $config
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8

Write-Output "Started OpenRQGM 128-outcome run: PID $($process.Id)"
Write-Output "Progress log: $stdout"

