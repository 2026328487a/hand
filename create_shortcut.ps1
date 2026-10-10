param([switch]$Desktop)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Python environment not found. See real\README.md.'
}
$taskPython = (Resolve-Path -LiteralPath $taskPython).Path
$taskEntry = Join-Path $taskRoot 'launch_gui.pyw'
if (-not (Test-Path -LiteralPath $taskEntry)) {
    throw "GUI launcher not found: $taskEntry"
}
$taskDestinations = @($taskRoot)
if ($Desktop) {
    $taskDestinations += [Environment]::GetFolderPath('Desktop')
}
$taskShell = New-Object -ComObject WScript.Shell
foreach ($taskFolder in $taskDestinations) {
    $taskLink = $taskShell.CreateShortcut((Join-Path $taskFolder '灵巧手控制台.lnk'))
    $taskLink.TargetPath = $taskPython
    $taskLink.Arguments = '"' + $taskEntry + '"'
    $taskLink.WorkingDirectory = $taskRoot
    $taskIcon = Join-Path $taskRoot 'app.ico'
    $taskLink.IconLocation = if (Test-Path -LiteralPath $taskIcon) { $taskIcon } else { "$taskPython,0" }
    $taskLink.Description = 'ROH-LiteS001 灵巧手控制台'
    $taskLink.WindowStyle = 1
    $taskLink.Save()
    Write-Output "Created: $($taskLink.FullName)"
}
