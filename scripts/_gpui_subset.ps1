Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-GpuiSubsetMembers {
    @(
        'crates/collections',
        'crates/refineable',
        'crates/refineable/derive_refineable',
        'crates/gpui',
        'crates/gpui_linux',
        'crates/gpui_macos',
        'crates/gpui_macros',
        'crates/gpui_platform',
        'crates/gpui_util',
        'crates/gpui_web',
        'crates/gpui_wgpu',
        'crates/gpui_windows',
        'crates/http_client',
        'crates/http_client_tls',
        'crates/media',
        'crates/reqwest_client',
        'crates/scheduler',
        'crates/sum_tree',
        'crates/util',
        'crates/util_macros',
        'crates/zlog',
        'crates/ztracing',
        'crates/ztracing_macro',
        'tooling/perf'
    )
}

function Get-GpuiRootFiles {
    @(
        '.cargo',
        'Cargo.toml',
        'Cargo.lock',
        'rust-toolchain.toml',
        'LICENSE-APACHE',
        'LICENSE-GPL',
        'README.md',
        '.gitignore'
    )
}

function Set-GpuiWorkspaceMembers {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CargoTomlPath
    )

    $membersBlock = @"
[workspace]
resolver = "2"
members = [
    "crates/collections",
    "crates/refineable",
    "crates/refineable/derive_refineable",
    "crates/gpui",
    "crates/gpui_linux",
    "crates/gpui_macos",
    "crates/gpui_macros",
    "crates/gpui_platform",
    "crates/gpui_util",
    "crates/gpui_web",
    "crates/gpui_wgpu",
    "crates/gpui_windows",
    "crates/http_client",
    "crates/http_client_tls",
    "crates/media",
    "crates/reqwest_client",
    "crates/scheduler",
    "crates/sum_tree",
    "crates/util",
    "crates/util_macros",
    "crates/zlog",
    "crates/ztracing",
    "crates/ztracing_macro",
    "tooling/perf",
]
default-members = ["crates/gpui"]
"@

    $content = Get-Content $CargoTomlPath -Raw
    $updated = [regex]::Replace(
        $content,
        '(?s)^\[workspace\].*?default-members = \[[^\]]*\]\r?\n',
        $membersBlock + "`r`n"
    )
    Set-Content -Path $CargoTomlPath -Value $updated -NoNewline
}