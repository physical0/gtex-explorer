param(
    [switch]$CheckOnly,
    [switch]$Headless,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$AppDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$Candidates = @()
if ($env:THESIS_PYTHON) { $Candidates += $env:THESIS_PYTHON }
$Candidates += Join-Path $AppDirectory ".venv\Scripts\python.exe"
$Candidates += Join-Path (Split-Path -Parent $AppDirectory) "venv\Scripts\python.exe"
$Candidates += Join-Path $AppDirectory "venv\Scripts\python.exe"
$SystemPython = Get-Command python -ErrorAction SilentlyContinue
if ($SystemPython) { $Candidates += $SystemPython.Source }

$PythonPath = $null
foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
    try {
        # A copied/moved venv can exist but refer to a missing base Python.
        # Verify the actual interpreter and all runtime imports, without a download.
        & $Candidate -c "import streamlit, pandas, numpy, altair, xgboost, sklearn, joblib; assert xgboost.__version__ == '3.2.0'" 2>$null
        if ($LASTEXITCODE -eq 0) { $PythonPath = $Candidate; break }
    } catch { continue }
}
if (-not $PythonPath) {
    throw "Tidak ada Python yang siap. Buat .venv di folder aplikasi dan instal requirements.txt, atau arahkan THESIS_PYTHON ke interpreter yang valid. Tidak ada paket yang diunduh otomatis."
}
Write-Output "Interpreter siap: $PythonPath"
if ($CheckOnly) { exit 0 }

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Set-Location -LiteralPath $AppDirectory
& $PythonPath -m streamlit run app.py --server.address=127.0.0.1 --server.port=$Port --server.headless=$($Headless.IsPresent.ToString().ToLower()) --browser.gatherUsageStats=false
exit $LASTEXITCODE
