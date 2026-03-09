# GPUI Fork Workflow

This repository is a standalone subset of the Zed monorepo focused on `gpui` and the internal crates it needs.

## Branches

- `upstream-main`: pure imports from `zed/crates/gpui` and its required workspace crates.
- `ame-graphics`: your working branch for graphics-specific patches.

Keep product changes on `ame-graphics`. Use `upstream-main` only for syncing from Zed.

## Local upstream cache

Scripts use a local full clone of Zed at `_upstream/zed` by default.

## Scripts

### Refresh the local Zed clone and sync the current branch

```powershell
./scripts/sync-from-zed.ps1
```

### Update `upstream-main` from Zed and return to your working branch

```powershell
./scripts/update-upstream-main.ps1
```

### Update `upstream-main` and merge it into `ame-graphics`

```powershell
./scripts/update-upstream-main.ps1 -MergeIntoWorkingBranch
```

## Suggested workflow

1. Work on `ame-graphics`.
2. When Zed moves, run `./scripts/update-upstream-main.ps1`.
3. Review the new sync commit on `upstream-main`.
4. Merge or rebase `ame-graphics` onto `upstream-main`.
5. Push your fork to GitHub and pin `ame` to a commit or branch.

## Notes

- This repository intentionally vendors the minimal workspace needed to build `gpui` outside the full Zed monorepo.
- `UPSTREAM_ZED_COMMIT` records the exact Zed commit that the current import came from.
- The root `Cargo.toml` keeps Zed's workspace dependency table but trims the active workspace members to the imported subset.