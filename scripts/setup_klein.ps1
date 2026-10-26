$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$env:TEMP = Join-Path $taskRoot '.tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $taskRoot '.cache/pip'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 to create the project GPU environment first.' }
# Resolve all dependencies in this project's dedicated Klein venv.
& $taskPython -m pip install -r (Join-Path $taskRoot 'requirements-klein.txt')
if ($LASTEXITCODE -ne 0) { throw 'Klein dependency installation failed.' }
& $taskPython (Join-Path $taskRoot 'scripts/download_klein.py')
if ($LASTEXITCODE -ne 0) { throw 'Klein download incomplete; rerun to resume.' }
