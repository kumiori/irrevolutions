# Branch Vitality Map

Date: 2026-03-26

This map classifies branches by:

- recency of last activity,
- commits in the last 90 / 365 days,
- divergence from `origin/main`,
- whether they appear to contain distinct scientific work rather than already-merged history.

Notation:

- `ahead/behind` is relative to `origin/main`
- vitality is a qualitative operational label, not a judgment on scientific value

## Active

### `main`

- last activity: 2026-03-26
- recent commits: 11 in 90d, 31 in 365d
- divergence: `0/0`
- vitality: `vital`

Notes:

- current development baseline
- now contains the bifurcation-tuning integration
- two latest merges are PR #118 and PR #119

### `andres-bifurcation-tuning`

- last activity: 2026-03-26
- recent commits: 9 in 90d, 29 in 365d
- divergence: `0/2`
- vitality: `merged-active-history`

Notes:

- recently active, but effectively superseded by `main`
- useful as historical context for the merge narrative
- not a branch to continue developing on

## Warm / Merge Candidate

### `andres-flow`

- last activity: 2025-06-24
- recent commits: 0 in 90d, 29 in 365d
- divergence: `21/23`
- vitality: `warm`

Signals:

- substantial unique work still ahead of `main`
- not catastrophically drifted
- recent commits mention `JumpSolver`, gradient fixes, and state-copy logic

Recommendation:

- highest-priority non-main branch to inspect next
- likely contains a coherent unfinished feature line rather than pure archive material

## Cool / Inspect If Scientifically Relevant

### `andres-new-models`

- last activity: 2025-05-27
- recent commits: 0 in 90d, 3 in 365d
- divergence: `0/28`
- vitality: `cool`

Signals:

- no unique commits ahead of `main`
- some relatively recent activity
- likely partially absorbed or abandoned

Recommendation:

- inspect only if model additions are specifically needed

### `andres-0.9`

- last activity: 2024-11-25
- recent commits: 0 in 90d, 0 in 365d
- divergence: `2/127`
- vitality: `cool-historical`

Signals:

- version-transition branch
- small amount of unique work, but heavily behind

Recommendation:

- keep only as a historical compatibility reference

## Dormant But Potentially Valuable

### `andres-practice`

- last activity: 2024-12-06
- recent commits: 0 in 90d, 0 in 365d
- divergence: `0/79`
- vitality: `dormant`

Signals:

- likely conceptually overlapped by the newly promoted `practice/`
- no remaining unique commits ahead of `main`

Recommendation:

- archive mentally; only mine for historical intent

### `andres-paper`

- last activity: 2024-12-05
- recent commits: 0 in 90d, 0 in 365d
- divergence: `0/139`
- vitality: `dormant`

Signals:

- paper-facing branch, not codebase-driving

Recommendation:

- inspect only for manuscript or figure recovery

### `andres-doc`

- last activity: 2024-09-24
- recent commits: 0 in 90d, 0 in 365d
- divergence: `23/203`
- vitality: `dormant-stranded`

Signals:

- documentation work exists ahead of `main`
- branch is very drifted

Recommendation:

- salvage documentation ideas manually instead of trying to merge the branch whole

### `andres-discrete`

- last activity: 2024-06-28
- recent commits: 0 in 90d, 0 in 365d
- divergence: `17/639`
- vitality: `dormant-stranded`

Signals:

- still has unique commits
- drift is severe

Recommendation:

- inspect only if discrete formulations become an immediate scientific priority

### `andres-linesearch`

- last activity: 2024-05-16
- recent commits: 0 in 90d, 0 in 365d
- divergence: `0/310`
- vitality: `dormant-absorbed`

Signals:

- no unique commits ahead of `main`
- historically important, operationally absorbed

Recommendation:

- no revival needed

### `andres-conerecipe`

- last activity: 2024-03-22
- recent commits: 0 in 90d, 0 in 365d
- divergence: `3/378`
- vitality: `dormant-stranded`

Signals:

- small amount of unique work
- very old cone/stability experimentation

Recommendation:

- inspect only for very specific cone-algorithm archaeology

### `andres-plates`

- last activity: 2024-03-09
- recent commits: 0 in 90d, 0 in 365d
- divergence: `25/457`
- vitality: `dormant-stranded`

Signals:

- nontrivial unique branch content
- heavily drifted

Recommendation:

- inspect only if plate models return to the roadmap

## Archive

Everything older than the branches above, especially branches with:

- no commits in the last year,
- no commits ahead of `main`,
- hundreds of commits behind `main`,

should be treated as archival unless there is a concrete scientific reason to revive them.

Examples:

- `andres-elasticity`
- `andres-meshes`
- `andres-viscous`
- `andres-notch`
- `andres-fatigue`
- older contributor branches from 2022 and earlier

## Recommended Next Inspection Order

If the goal is to continue research development from the newly merged `main`, inspect branches in this order:

1. `andres-flow`
2. `andres-new-models`
3. `andres-doc`
4. `andres-discrete`
5. `andres-plates`

## Short Conclusion

The repository now has one genuinely live branch: `main`.

`andres-bifurcation-tuning` should now be thought of as integrated history, not an active line.

The only branch that currently looks both distinct and plausibly revivable without extreme archaeology is `andres-flow`.
