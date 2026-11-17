$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root

if (-not (Test-Path "cp.txt")) {
    mvn -q dependency:build-classpath "-Dmdep.outputFile=cp.txt"
    if ($LASTEXITCODE -ne 0) { throw "classpath build failed" }
}
mvn -q compile
if ($LASTEXITCODE -ne 0) { throw "compile failed" }

$cp = "target\classes;" + (Get-Content cp.txt -Raw).Trim()
New-Item -ItemType Directory -Force -Path results | Out-Null
$out = "results\chunking-m3.txt"
if (Test-Path $out) { Remove-Item $out }

# order matters: sweep tunes on train, the rest use the locked configs on test
$runs = @("RunChunkingSweep", "RunChunkingFinal", "RunChunkingSignificance", "RunChunkingTitleAblation")
foreach ($r in $runs) {
    "##### $r #####" | Tee-Object -FilePath $out -Append
    java -cp $cp "io.github.abdullah.rag.longdoc.$r" | Tee-Object -FilePath $out -Append
    if ($LASTEXITCODE -ne 0) { throw "$r failed" }
}
Write-Host "Done. Results saved to $out"