$ErrorActionPreference = 'Stop'
$url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
$dest = Join-Path $PSScriptRoot '..\data'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = Join-Path $dest 'scifact.zip'
if (-not (Test-Path (Join-Path $dest 'scifact'))) {
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dest
    Remove-Item $zip
}
Write-Host 'SciFact ready in' $dest
