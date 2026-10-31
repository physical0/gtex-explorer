param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$AppDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $AppDirectory '.venv\Scripts\python.exe'
if ($env:THESIS_PYTHON) { $PythonPath = $env:THESIS_PYTHON }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Environment desktop belum tersedia. Instal requirements-desktop.txt atau gunakan executable hasil build. Tidak ada unduhan otomatis.'
}
$Arguments = @((Join-Path $AppDirectory 'desktop.py'))
if ($CheckOnly) { $Arguments += '--check' }
& $PythonPath @Arguments
exit $LASTEXITCODE
