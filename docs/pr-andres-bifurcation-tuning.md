# PR Draft: Bifurcation Tuning, Practice Promotion, and Runtime Compatibility

## Title

`Bifurcation tuning, practice script promotion, and DOLFINx 0.9 runtime fixes`

## Summary

This PR consolidates the recent work on bifurcation/stability tuning and brings the repository back to a runnable state on DOLFINx 0.9.

It does three things at once:

1. extends the first- and second-order solver stack used by the 1D bifurcation-tuning workflow,
2. promotes the old `src/irrevolutions/practice/` scripts into a root-level `practice/` application layer,
3. removes accumulated runtime drift across demos, tests, and practice scripts.

The result is a clearer repository split:

- `demo/`: canonical applications
- `practice/`: serious research workflows expected to run
- `playground/`: branch-local exploration and tuning

## Why

Before this work, the branch contained valuable solver work, but execution had drifted badly:

- editable installs and packaging metadata were inconsistent,
- many scripts still used stale DOLFINx APIs,
- plotting helpers had headless/runtime breakage,
- several practice scripts relied on legacy solver call patterns or broken path assumptions,
- the old practice layer lived inside `src/`, where it no longer matched its role.

This PR restores operational coherence without trying to redesign the scientific logic.

## Main Changes

### 1. Solver and model work

- extends the bifurcation/stability workflow used by `playground/bif-tuning/1d-traction-tuning.py`,
- improves `src/irrevolutions/algorithms/am.py` and `src/irrevolutions/algorithms/so.py`,
- adds backward-compatible solver aliases still used by older scripts,
- adds missing/default stability-eigensolver handling needed by legacy parameter files,
- expands constitutive/model support in `src/irrevolutions/models/__init__.py` and `src/irrevolutions/models/one_dimensional.py`.

### 2. Practice promotion

- creates a root-level `practice/` directory,
- promotes the old `src/irrevolutions/practice/*.py` entry points into that directory,
- adds stable root/path helpers for runtime data and parameter resolution,
- keeps the old `src/irrevolutions/practice/` copies in place for transition safety.

### 3. DOLFINx 0.9 compatibility fixes

- migrates old `FunctionSpace(...)` construction to `functionspace(...)`,
- fixes interpolation, refinement, and function update call sites that drifted with DOLFINx changes,
- updates scripts that relied on older PETSc/SNES or BC object behaviour,
- repairs stale script-side solver calls by moving compatibility logic to the application edge instead of the solver core.

### 4. Visualisation and headless runtime fixes

- normalises plotting behaviour in `src/irrevolutions/utils/viz.py`,
- restores compatibility with older code that expects plot helpers to behave like plotters,
- adds safer offscreen/screenshot behaviour for headless runs,
- fixes Matplotlib/PyVista return-shape mismatches that were breaking tests and demos.

### 5. Packaging and repo hygiene

- normalises package metadata in `pyproject.toml`,
- removes a broken console entry point,
- fixes timing-table behaviour when timers are absent,
- adds helper tooling and docs to validate the promoted practice layer.

## Validation

### Demos

The key demo layer was exercised and patched against current APIs:

- `demo_bifurcation.py`
- `demo_contact.py`
- `demo_elasticity.py`
- `demo_traction.py`
- `demo_vi.py`

### Practice matrix

A dedicated runner was added:

- `tools/run_practice_matrix.py`
- `tools/show_practice_matrix_failures.py`

This was used to iteratively rehabilitate the promoted `practice/` scripts.

At the end of the debugging cycle, the matrix had reached:

- all promoted practice scripts passing, except `default.py` before the final output-directory patch,
- the final `default.py` patch addresses that last concrete failure mode.

Recommended final verification command:

```bash
python3 tools/run_practice_matrix.py --show-output --failure-lines 80
```

## Reviewer Guide

Review by theme rather than by file order.

### Review Pass 1: solver/core changes

- `src/irrevolutions/algorithms/am.py`
- `src/irrevolutions/algorithms/so.py`
- `src/irrevolutions/models/__init__.py`
- `src/irrevolutions/models/one_dimensional.py`

Focus:

- first-order / second-order handoff,
- inertia/eigensolver behaviour,
- compatibility changes that affect scientific behaviour.

### Review Pass 2: runtime compatibility layer

- `src/irrevolutions/utils/viz.py`
- `src/irrevolutions/utils/plots.py`
- `src/irrevolutions/utils/__init__.py`
- `src/irrevolutions/utils/compat.py`

Focus:

- API drift fixes,
- headless plotting behaviour,
- whether compatibility logic stays at the boundary rather than in the core.

### Review Pass 3: application surface

- `practice/`
- updated `demo/` scripts
- `playground/bif-tuning/1d-traction-tuning.py`

Focus:

- runnable workflows,
- path/data conventions,
- whether practice/demo/playground roles are now clearer.

## Risks / Follow-Up

This PR intentionally fixes operational drift without fully redesigning the application architecture.

Likely follow-up work:

- formalise a minimal experiment contract (`state`, `bcs`, `bounds`, history/results),
- introduce a reusable experiment runner for the repeated load-step choreography,
- continue moving compatibility shims out of the core and into application-facing helpers,
- decide whether to retire or further curate historical playground material.

## Suggested PR Description

This PR restores the branch to a runnable research state on DOLFINx 0.9 whilst consolidating the recent bifurcation-tuning work.

The main outcomes are:

- the 1D bifurcation/stability tuning workflow is integrated into the branch,
- the old `src/irrevolutions/practice/` scripts are promoted into a root `practice/` application layer,
- demos, tests, and practice scripts are updated for current DOLFINx and headless plotting behaviour,
- packaging/runtime inconsistencies that were causing stale imports and broken entry points are cleaned up,
- a practice-matrix runner is added so the promoted workflows can be checked systematically.

The PR is broad, but the changes fall into four reviewable buckets: solver/model evolution, runtime/API compatibility, practice promotion, and validation tooling.
