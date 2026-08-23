$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$dockerfile = Join-Path $PSScriptRoot "Dockerfile.polyglot"

foreach ($language in @("cpp", "go", "java", "javascript", "python", "rust")) {
    $tag = "openrqgm-polyglot-$language`:2026-08-23"
    docker build --target $language -t $tag -f $dockerfile $root
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build $tag"
    }
}
