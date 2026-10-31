param([string]$DistributionDirectory = 'dist')
$ErrorActionPreference = 'Stop'
$AppDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $AppDirectory '.venv\Scripts\python.exe'
if ($env:THESIS_PYTHON) { $PythonPath = $env:THESIS_PYTHON }
Push-Location -LiteralPath $AppDirectory
try {
    # Dependencies must be installed beforehand. Building never runs pip or downloads a model.
    & $PythonPath desktop.py --check
    if ($LASTEXITCODE -ne 0) { throw 'Pemeriksaan model dan data gagal.' }
    & $PythonPath -m PyInstaller desktop.spec --noconfirm --distpath $DistributionDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Build desktop gagal.' }
    # Test the bundled runtime, not only the source environment.
    $PackageDirectory = Join-Path $DistributionDirectory 'GTExTissueExplorer'
    $PackagedCheck = Start-Process -FilePath (Join-Path $PackageDirectory 'GTExTissueExplorer.exe') -ArgumentList '--check' -WindowStyle Hidden -PassThru -Wait
    if ($PackagedCheck.ExitCode -ne 0) { throw 'Pemeriksaan executable gagal. Periksa desktop.log dan last_error.txt.' }
    Copy-Item -LiteralPath (Join-Path $AppDirectory 'DESKTOP.md') -Destination (Join-Path $PackageDirectory 'CARA_MENJALANKAN.md')
    Write-Output "Selesai: $PackageDirectory\GTExTissueExplorer.exe"
} finally { Pop-Location }
