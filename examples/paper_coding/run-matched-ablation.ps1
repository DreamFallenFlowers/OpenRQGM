param(
    [ValidateSet(512, 1024)]
    [int]$Budget,
    [ValidateSet("verifier_only", "fixed_reviewer", "coevolving_reviewer")]
    [string]$Condition,
    [switch]$Preflight,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "ablation.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "OpenRQGM virtual environment is missing: $python"
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $root "src"

if ($Preflight -or -not $Budget -or -not $Condition) {
    & $python $runner --preflight
    exit $LASTEXITCODE
}

docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon is not ready."
}
foreach ($language in @("cpp", "go", "java", "javascript", "python", "rust")) {
    docker image inspect "openrqgm-polyglot-$language`:2026-08-23" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing image for $language; run build-images.ps1 first."
    }
}

& $python $runner --preflight | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Matched-ablation preflight failed."
}

$arguments = @($runner, "--run", "--budget", "$Budget", "--condition", $Condition)
$runs = Join-Path $root "runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
foreach ($existingFile in Get-ChildItem -LiteralPath $runs -Filter "ablation-*-process.json") {
    try {
        $existing = Get-Content -LiteralPath $existingFile.FullName -Raw | ConvertFrom-Json
    }
    catch {
        continue
    }
    $active = Get-Process -Id $existing.pid -ErrorAction SilentlyContinue
    if ($active) {
        throw "Another matched-ablation cell is active: PID $($existing.pid)"
    }
}
if ($Foreground) {
    & $python @arguments
    exit $LASTEXITCODE
}

$name = "ablation-$Budget-$Condition"
$stdout = Join-Path $runs "$name-console.log"
$stderr = Join-Path $runs "$name-error.log"
$processFile = Join-Path $runs "$name-process.json"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    budget = $Budget
    condition = $Condition
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8

Write-Output "Started matched ablation: budget=$Budget condition=$Condition PID=$($process.Id)"
Write-Output "Progress log: $stdout"
