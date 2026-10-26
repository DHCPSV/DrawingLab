param([string]$Python = 'python', [switch]$WithModels,
      [ValidateSet('klein','counterfeit')][string]$Engine = 'klein')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path '.tmp','.cache' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & $Python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), 'Use Python 3.11 or 3.12; pass -Python with its executable path.'"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11/3.12 is required.' }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $pythonPath -m pip install --disable-pip-version-check -r requirements.txt -r requirements-test.txt
if ($LASTEXITCODE -ne 0) { throw 'Core dependencies failed.' }
if ($WithModels) {
    & $pythonPath -m pip install --disable-pip-version-check torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    if ($Engine -eq 'klein') {
        & (Join-Path $PSScriptRoot 'scripts/setup_klein.ps1')
    } else {
        & $pythonPath -m pip install --disable-pip-version-check -r requirements-gpu.txt
        if ($LASTEXITCODE -ne 0) { throw 'Model dependencies failed.' }
        & $pythonPath scripts/download_models.py
        if ($LASTEXITCODE -ne 0) { throw 'Model download failed; rerun to resume.' }
    }
    New-Item -ItemType Directory -Force -Path 'models' | Out-Null
    @{ engine = $Engine } | ConvertTo-Json | Set-Content -Encoding UTF8 'models/active-engine.json'
}
Write-Host 'Ready. Run ./start.ps1, then open http://127.0.0.1:8765'
