# Practice Migration Matrix

The promoted practice layer now lives in [practice](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice).
Legacy copies under `src/irrevolutions/practice` are still present during the transition, but the intended runnable entrypoints are the root-level scripts below.

Use [tools/run_practice_matrix.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/tools/run_practice_matrix.py) to execute them one by one against the editable source tree. The runner prepends `src/` to `PYTHONPATH`, sets `MPLBACKEND=Agg`, and writes a JSON report to `output/practice-script-matrix.json` by default.

## Commands

```bash
python3 tools/run_practice_matrix.py --list
python3 tools/run_practice_matrix.py --timeout 300
python3 tools/run_practice_matrix.py --only traction_bar_clean
python3 tools/run_practice_matrix.py --include-launchers
```

## Matrix

| Script | Kind | Parameters | Migration Notes |
|---|---|---|---|
| `practice/default.py` | application | `test/parameters.yml` | Promoted, imports normalized, debugger trap removed. |
| `practice/discrete_atk.py` | application | `practice/parameters.yml` | Promoted, imports normalized, local parameter path stabilized. |
| `practice/discrete_atk_homogeneous.py` | application | `practice/parameters.yml` | Promoted, imports normalized, local parameter path stabilized, debugger trap removed. |
| `practice/enpassant.py` | application | internal | Promoted, imports normalized. Still exploratory in shape. |
| `practice/multiaxial-disc.py` | application | `test/parameters.yml` | Promoted, parameter path stabilized. |
| `practice/pacman-cone.py` | application | generated/written under output dir | Promoted, imports normalized. |
| `practice/pacman_hybrid.py` | application | generated/written under output dir | Promoted, imports normalized. |
| `practice/thinfilm-bar.py` | application | `data/thinfilm/parameters.yml` | Promoted, imports normalized, data path stabilized. |
| `practice/traction-AT1_cone.py` | application | `test/parameters.yml` | Promoted, imports normalized, parameter path stabilized. |
| `practice/traction-AT1_first_order.py` | application | `test/parameters.yml` | Promoted, imports normalized, parameter path stabilized. |
| `practice/traction-AT2_cone.py` | application | `test/parameters.yml` | Promoted, imports normalized, `so_merged` dependency replaced with `irrevolutions.algorithms.so`. |
| `practice/traction-ATJJ.py` | application | `test/parameters.yml`, `test/atk_parameters.yml` | Promoted, imports normalized, parameter paths stabilized. |
| `practice/traction-bar-clean.py` | application | `test/parameters.yml` | Promoted, imports normalized, parameter path stabilized. |
| `practice/traction-cone.py` | application | `test/parameters.yml` | Promoted, imports normalized, parameter path stabilized. |
| `practice/traction-parametric.py` | application | `test/parameters.yml` | Promoted, imports normalized, parameter path stabilized. |
| `practice/unstabinst.py` | application | internal | Promoted, imports normalized. Still exploratory in shape. |
| `practice/parametric-traction-bar-r.py` | launcher | none | Helper sweep launcher for `traction-parametric.py`. Excluded from default runner set. |
| `practice/parametric-traction-bar-s.py` | launcher | none | Helper sweep launcher for `traction-parametric.py`. Excluded from default runner set. |

## Current Scope

This migration pass intentionally does not redesign the scientific logic.
It does four narrower things:

- promotes the practice entrypoints to a root-level `practice/` layer,
- normalizes imports toward `irrevolutions.*`,
- stabilizes path lookups against the new location,
- provides a repeatable runner so the migrated set can be validated empirically.

The next phase can build on this by replacing repeated solve loops with a shared experiment runner and by introducing the standardized equilibrium object you want between first-order and second-order analysis.
