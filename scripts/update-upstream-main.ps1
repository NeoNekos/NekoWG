param(
    [string]$ZedPath = (Join-Path $PSScriptRoot '..\_upstream\zed'),
    [string]$Ref = 'origin/main',
    [switch]$MergeIntoWorkingBranch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path $PSScriptRoot -Parent
$currentBranch = (git -C $repoRoot branch --show-current).Trim()
if (-not $currentBranch) {
    throw 'Repository is in detached HEAD state.'
}

$workingBranch = if ($currentBranch -eq 'upstream-main') { 'ame-graphics' } else { $currentBranch }

git -C $repoRoot checkout upstream-main | Out-Null
& (Join-Path $PSScriptRoot 'sync-from-zed.ps1') -ZedPath $ZedPath -Ref $Ref -CreateCommit

if ($MergeIntoWorkingBranch -and $workingBranch -ne 'upstream-main') {
    git -C $repoRoot checkout $workingBranch | Out-Null
    git -C $repoRoot merge --no-ff upstream-main
} else {
    git -C $repoRoot checkout $workingBranch | Out-Null
}