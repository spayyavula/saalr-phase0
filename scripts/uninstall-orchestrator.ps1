<#
.SYNOPSIS
    Remove the Saalr Phase 0 orchestrator Scheduled Task. Leaves
    .phase0/state.json on disk so the pipeline can be resumed by
    re-installing the task later.
#>
[CmdletBinding()]
param(
    [string]$TaskName = "Saalr Phase 0 Orchestrator"
)

$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "No scheduled task named '$TaskName' is registered. Nothing to do."
    exit 0
}

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Unregistered '$TaskName'. State in .phase0\ is untouched."
