param([switch]$Desktop)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskEnv = Join-Path $taskRoot '.venv'
$taskPython = Join-Path $taskEnv 'Scripts\python.exe'
$taskVersion = (Get-Content -LiteralPath (Join-Path $taskRoot '.python-version') -Raw).Trim()

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install it with: python -m pip install --user uv; then reopen the terminal.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'ohand_serial_sdk_python\setup.py'))) {
    throw 'Missing ohand_serial_sdk_python. Restore the SDK source as described in README.md.'
}

Push-Location $taskRoot
try {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        if (Test-Path -LiteralPath $taskEnv) {
            throw "Incomplete environment at $taskEnv. Rename it before retrying."
        }
        & uv venv --python $taskVersion $taskEnv
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
    }
    $taskActual = & $taskPython -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0 -or $taskActual -ne $taskVersion) {
        throw "Expected Python $taskVersion at $taskPython. Rename the existing .venv before retrying."
    }

    & uv pip sync --link-mode copy --python $taskPython requirements-gui.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & uv pip check --python $taskPython
    if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
    & $taskPython (Join-Path $taskRoot 'check_env.py') --core-only
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed. See the output above.' }

    & (Join-Path $taskRoot 'create_shortcut.ps1') -Desktop:$Desktop
    Write-Host "Ready. Python: $taskPython"
    Write-Host 'Open the shortcut in real/ to launch the GUI.'
} finally {
    Pop-Location
}
