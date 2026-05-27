<#
.SYNOPSIS
    Register the Saalr Phase 0 orchestrator as a Windows Scheduled Task.

.DESCRIPTION
    Creates a per-user Scheduled Task that runs `python -m src.orchestrator
    run` at logon, restarts on failure, and works while the laptop is on
    battery. State persists in <repo>\.phase0\state.json; the orchestrator
    picks up where it left off on every restart (sleep/wake, reboot,
    Ctrl+C).

    Re-run with -Force to update the task in place.

.PARAMETER PythonPath
    Path to python.exe. Defaults to the first `python` on PATH.

.PARAMETER RepoRoot
    Repo root. Defaults to the parent of the script directory.

.PARAMETER TaskName
    Scheduled Task name. Defaults to "Saalr Phase 0 Orchestrator".

.EXAMPLE
    .\scripts\install-orchestrator.ps1
    .\scripts\install-orchestrator.ps1 -PythonPath C:\Python311\python.exe -Force
#>
[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$RepoRoot,
    [string]$TaskName = "Saalr Phase 0 Orchestrator",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
}
if (-not (Test-Path (Join-Path $RepoRoot "pre-registration.md"))) {
    throw "RepoRoot $RepoRoot does not contain pre-registration.md"
}

if (-not $PythonPath) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "python not found on PATH; pass -PythonPath explicitly"
    }
    $PythonPath = $cmd.Source
}
if (-not (Test-Path $PythonPath)) {
    throw "PythonPath $PythonPath does not exist"
}

Write-Host "Installing scheduled task '$TaskName'"
Write-Host "  RepoRoot: $RepoRoot"
Write-Host "  Python:   $PythonPath"

$action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "-m src.orchestrator run" `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -RestartCount 3 `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    if (-not $Force) {
        throw "Task '$TaskName' already exists. Re-run with -Force to replace."
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$description = @"
Saalr Phase 0 background pipeline (backfill → sentiment → IV → matching →
baselines → validation eval). State lives in $RepoRoot\.phase0\state.json.
Logs to $RepoRoot\.phase0\orchestrator.log.

Manage: python -m src.orchestrator {status|pause|resume|reset STAGE}
Uninstall: scripts\uninstall-orchestrator.ps1
"@

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description $description | Out-Null

Write-Host ""
Write-Host "Installed. The orchestrator will start 30 s after your next logon."
Write-Host "Start it now without rebooting:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Check status:"
Write-Host "  python -m src.orchestrator status"
