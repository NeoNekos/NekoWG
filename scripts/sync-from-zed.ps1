param(
    [string]$ZedPath = (Join-Path $PSScriptRoot '..\_upstream\zed'),
    [string]$ZedRemote = 'https://github.com/zed-industries/zed.git',
    [string]$Ref = 'origin/main',
    [switch]$SkipFetch,
    [switch]$CreateCommit,
    [string]$CommitMessage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '_gpui_subset.ps1')

$repoRoot = Split-Path $PSScriptRoot -Parent
$members = Get-GpuiSubsetMembers
$rootFiles = Get-GpuiRootFiles

New-Item -ItemType Directory -Force -Path (Split-Path $ZedPath -Parent) | Out-Null
if (Test-Path $ZedPath) {
    if (-not $SkipFetch) {
        git -C $ZedPath fetch origin --tags --prune
    }
} else {
    git clone $ZedRemote $ZedPath
}

git -C $ZedPath checkout --detach $Ref | Out-Null
$upstreamCommit = (git -C $ZedPath rev-parse HEAD).Trim()
Write-Host "Syncing from zed commit $upstreamCommit"

foreach ($path in $rootFiles) {
    $src = Join-Path $ZedPath $path
    $dst = Join-Path $repoRoot $path
    if (Test-Path $dst) {
        Remove-Item -Recurse -Force $dst
    }
    if (Test-Path $src) {
        $parent = Split-Path $dst -Parent
        if ($parent) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }
        Copy-Item -Recurse -Force $src $dst
    }
}

foreach ($area in @('crates', 'tooling')) {
    $target = Join-Path $repoRoot $area
    if (Test-Path $target) {
        Get-ChildItem $target -Force | Remove-Item -Recurse -Force
    } else {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
    }
}

foreach ($member in $members) {
    $src = Join-Path $ZedPath $member
    if (-not (Test-Path $src)) {
        throw "Missing upstream path: $member"
    }
    $dst = Join-Path $repoRoot $member
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -Recurse -Force $src $dst
}

Set-GpuiWorkspaceMembers -CargoTomlPath (Join-Path $repoRoot 'Cargo.toml')
Set-Content -Path (Join-Path $repoRoot 'UPSTREAM_ZED_COMMIT') -Value $upstreamCommit

cargo metadata --format-version 1 --manifest-path (Join-Path $repoRoot 'Cargo.toml') > $null

if ($CreateCommit) {
    git -C $repoRoot add -A
    if (-not $CommitMessage) {
        $CommitMessage = "sync: zed $upstreamCommit"
    }
    git -C $repoRoot commit -m $CommitMessage
}

Write-Host 'Sync complete.'