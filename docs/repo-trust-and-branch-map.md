# Repo Trust And Branch Map

This document is a working trust map for the Python entry points and tests in this repository, plus a branch activity map built from local git history.

The goal is pragmatic:

- identify where to start running code,
- separate package-backed reference code from exploratory drift,
- highlight known breakpoints before time is wasted,
- give a branch-level view of where the work actually happened.

## Method

Signals used for the trust map:

- package imports from `irrevolutions`,
- bare imports such as `from algorithms...`, `from meshes...`, `from models...`,
- `sys.path.append(...)` path hacks,
- active debugger traps,
- `__main__` guards or `main()` entry points,
- obvious missing references such as `algorithms.so_merged`,
- per-file commit counts and last touching commit,
- syntax-only parsing pass across every Python file listed here.

Important limits:

- this is not a runtime guarantee,
- the current machine does not have the FEniCS stack installed, so DOLFINx-dependent scripts/tests were not executed here,
- a file can be syntactically valid and still be operationally stale.

## Trust Scale

- `A` Reference: packaged or low-drift entry point; this is where runs should start.
- `B` Runnable with caveats: likely usable, but depends on environment, cwd, or incomplete maintenance.
- `C` Exploratory/archival: research code, path-coupled scripts, or tests with brittle wiring.
- `D` Known broken/stub: missing references, empty files, or obviously incomplete entry points.

## High-Level Conclusions

- The real library core is still [src/irrevolutions](/Users/kumiori3/Documents/WIP/Nature/code_mec647/src/irrevolutions). The most trustworthy run targets are the demos and a small subset of `practice/`.
- Most `practice/` scripts are conceptually current but operationally coupled to old execution habits: bare imports plus `sys.path` injection. They are not clean installed-package entry points.
- `playground/` contains a mix of useful benchmarks and archival material. It is not uniformly stale, but it is not a safe default.
- One practice script is clearly broken by construction: [src/irrevolutions/practice/traction-AT2_cone.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/src/irrevolutions/practice/traction-AT2_cone.py) imports `algorithms.so_merged`, which is absent.
- The branch history shows recent work concentrated on `main`, `andres-flow`, and `andres-bifurcation-tuning`, with a long tail of dormant feature branches containing unmerged experiments.

## First Run Queue

If you start running the scripts you remember should work, I would begin in this order:

1. `demo/demo_elasticity.py`
2. `demo/demo_vi.py`
3. `demo/demo_bifurcation.py`
4. `demo/demo_traction.py`
5. `src/irrevolutions/practice/traction-bar-clean.py`
6. `src/irrevolutions/practice/thinfilm-bar.py`
7. `src/irrevolutions/practice/traction-parametric.py`
8. `playground/bif-tuning/1d-traction-tuning.py`

That sequence moves from package-backed demos into the strongest current research workflows.

## Demo

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `demo/demo_elasticity.py` | `A` | 2024-12-09 `c361eab` | 9 | pkg |
| `demo/demo_vi.py` | `A` | 2024-11-26 `172ad77` | 7 | pkg |
| `demo/demo_bifurcation.py` | `A` | 2024-09-05 `8d3e41f` | 8 | pkg |
| `demo/demo_contact.py` | `A` | 2025-03-06 `0cf96e2` | 1 | pkg |
| `demo/demo_traction.py` | `A` | 2025-03-06 `0cf96e2` | 10 | pkg |
| `demo/demo_biharmonic.py` | `A` | 2025-03-06 `014f147` | 1 | pkg |
| `demo/demo_strong_aniso.py` | `B` | untracked `0` | 0 | untracked |
| `demo/demo_vector_info.py` | `B` | 2024-08-13 `6f57ea7` | 2 | `__main__`, no package import |

## Practice

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `src/irrevolutions/practice/traction-bar-clean.py` | `B` | 2024-12-09 `c361eab` | 16 | pkg, bare-imports, sys.path, `__main__`, `main()` |
| `src/irrevolutions/practice/thinfilm-bar.py` | `B` | 2024-12-06 `7820a15` | 16 | pkg, bare-imports, sys.path, `__main__`, `main()` |
| `src/irrevolutions/practice/traction-parametric.py` | `B` | 2024-12-09 `c361eab` | 17 | pkg, bare-imports, sys.path, `__main__`, `main()` |
| `src/irrevolutions/practice/traction-cone.py` | `B` | 2024-12-09 `c361eab` | 17 | pkg, bare-imports, sys.path, `__main__` |
| `src/irrevolutions/practice/pacman-cone.py` | `B` | 2024-12-06 `7820a15` | 16 | pkg, bare-imports, sys.path, `__main__` |
| `src/irrevolutions/practice/pacman_hybrid.py` | `B` | 2024-12-06 `7820a15` | 16 | pkg, bare-imports, sys.path, `__main__` |
| `src/irrevolutions/practice/traction-AT1_cone.py` | `B` | 2024-12-06 `7820a15` | 14 | pkg, bare-imports, sys.path |
| `src/irrevolutions/practice/traction-AT1_first_order.py` | `B` | 2024-12-06 `7820a15` | 14 | pkg, bare-imports, sys.path |
| `src/irrevolutions/practice/traction-ATJJ.py` | `B` | 2024-12-09 `c361eab` | 18 | pkg, bare-imports, sys.path, `__main__` |
| `src/irrevolutions/practice/default.py` | `B` | 2024-09-05 `8d3e41f` | 14 | pkg, bare-imports, sys.path, debugger, `__main__` |
| `src/irrevolutions/practice/discrete_atk.py` | `B` | 2024-09-05 `8d3e41f` | 16 | pkg, bare-imports, `__main__` |
| `src/irrevolutions/practice/discrete_atk_homogeneous.py` | `B` | 2024-09-05 `8d3e41f` | 16 | pkg, bare-imports, debugger, `__main__` |
| `src/irrevolutions/practice/enpassant.py` | `C` | 2024-12-09 `c361eab` | 13 | bare-imports, sys.path |
| `src/irrevolutions/practice/multiaxial-disc.py` | `C` | 2024-09-03 `255062e` | 10 | sys.path, `__main__` |
| `src/irrevolutions/practice/parametric-traction-bar-r.py` | `C` | 2024-08-13 `6f57ea7` | 4 | plain helper script |
| `src/irrevolutions/practice/parametric-traction-bar-s.py` | `C` | 2024-08-13 `6f57ea7` | 4 | plain helper script |
| `src/irrevolutions/practice/unstabinst.py` | `C` | 2024-12-09 `c361eab` | 15 | bare-imports, sys.path |
| `src/irrevolutions/practice/traction-AT2_cone.py` | `D` | 2024-12-06 `7820a15` | 14 | pkg, bare-imports, sys.path, missing `algorithms.so_merged` |

## Playground

### Stronger Playground Targets

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `playground/bif-tuning/1d-traction-tuning.py` | `B` | 2025-07-01 `65e7f3a` | 5 | pkg, `__main__` |
| `playground/brazilian/brazilian.py` | `B` | 2025-02-25 `060aa8a` | 1 | pkg, `__main__` |
| `playground/pizza-notch/pizza-notch.py` | `B` | 2024-11-26 `172ad77` | 13 | pkg, `__main__` |
| `playground/rayleigh/rayleigh.py` | `B` | 2024-10-08 `f71b459` | 2 | pkg, `__main__` |
| `playground/benchmark-umut-at2/kicking_the_door.py` | `B` | 2024-12-09 `c361eab` | 3 | pkg, `__main__` |
| `playground/benchmark-umut-at2/vs_analytics_at2.py` | `B` | 2024-12-06 `7820a15` | 17 | pkg, `__main__` |
| `playground/benchmark-umut-at2/vs_analytics_at2_2d.py` | `B` | 2024-12-06 `7820a15` | 14 | pkg, `__main__` |
| `playground/benchmark-umut-at2/vs_analytics_at2_crunch_only.py` | `B` | 2024-11-26 `172ad77` | 2 | pkg, `__main__` |

### IDRIS Campaign Scripts

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-second-order-bifurcation.py` | `B` | 2024-11-26 `172ad77` | 5 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-second-order-stab-kick.py` | `B` | 2024-12-06 `7820a15` | 10 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-second-order-stab.py` | `B` | 2024-11-26 `172ad77` | 5 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-second-order-bifurcation-natural.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-first-order-new-hybrid-redundant.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-first-order-new-hybrid-natural.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-film-first-order-legacy-algorithm.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-traction-first-order-new-hybrid.py` | `B` | 2024-11-26 `172ad77` | 4 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-traction-first-order-legacy-algorithm.py` | `B` | 2024-11-26 `172ad77` | 5 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/1d-traction-first-order.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/2d-film-first-order-new-algorithm.py` | `B` | 2024-11-26 `172ad77` | 6 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/2d-film-second-order-bifurcation.py` | `B` | 2024-11-26 `172ad77` | 4 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/2d-film-second-order-stab-kick.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/2d-film-second-order-stability.py` | `B` | 2024-11-26 `172ad77` | 3 | pkg, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/libidris/core.py` | `C` | 2024-05-16 `9642a44` | 2 | support module, not an entry point |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/libidris/__init__.py` | `D` | 2024-05-15 `05bc4c8` | 1 | empty helper module |
| `playground/IDRIS-CAMPAING-HYDRA/scripts/2d-film-first-order-legacy-algorithm.py` | `D` | 2024-05-16 `22b46d0` | 1 | empty file |

### Lower-Trust Playground Areas

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `playground/rayleigh/rayleigh_parametric.py` | `C` | 2024-11-26 `b3c8095` | 3 | pkg, sys.path, `__main__` |
| `playground/2dcracks/2d-film-second-order-stab-kick.py` | `C` | untracked `0` | 0 | untracked, `__main__` |
| `playground/2dcracks/2d-film-second-order-stability.py` | `C` | untracked `0` | 0 | untracked, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/sandbox/1D/_1d_review.py` | `C` | untracked `0` | 0 | untracked, `__main__` |
| `playground/IDRIS-CAMPAING-HYDRA/sandbox/1D/_old_traction-first_order.py` | `C` | untracked `0` | 0 | pkg, bare-imports, sys.path |
| `playground/nb/visuals.py` | `C` | 2024-08-13 `e5f78a0` | 5 | helper module, not a runnable workflow |

## Test

### Highest-Confidence Lightweight Tests

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `test/test_meta.py` | `A` | 2024-08-13 `3599a65` | 2 | pure Python |
| `test/test_variational_iterator.py` | `A` | 2024-08-13 `6f57ea7` | 3 | pure Python |
| `test/test_errorcodes.py` | `A` | 2024-08-13 `3599a65` | 4 | `__main__`, no DOLFINx import |
| `test/test_logging_mpi.py` | `A` | 2024-08-13 `6f57ea7` | 6 | `__main__`, no DOLFINx import |

### Package-Oriented FEM Tests

These look current enough to matter, but they are environment-gated and were not runnable here because `dolfinx`/`ufl` are missing.

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `test/test_1d.py` | `B` | 2025-06-19 `c26f1d8` | 36 | pkg, `__main__` |
| `test/test_linsearch.py` | `B` | 2025-03-06 `67ce3b2` | 31 | pkg, `__main__` |
| `test/test_splits.py` | `B` | 2025-02-25 `1da677b` | 2 | pkg |
| `test/test_cone_convergence.py` | `B` | 2024-09-05 `8d3e41f` | 8 | pkg |
| `test/test_extend.py` | `B` | 2024-09-03 `00a2e66` | 12 | pkg |

### Brittle or Path-Coupled Tests

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `test/test_cone_project.py` | `C` | 2024-09-03 `255062e` | 17 | pkg, sys.path, `__main__` |
| `test/test_restriction.py` | `C` | 2024-09-03 `00a2e66` | 15 | pkg, sys.path, `__main__` |
| `test/test_sample_data.py` | `C` | 2024-09-05 `8d3e41f` | 9 | pkg, sys.path, `__main__` |
| `test/test_scatter.py` | `C` | 2024-09-05 `8d3e41f` | 11 | pkg, sys.path |
| `test/test_binarydataio.py` | `C` | 2024-09-03 `7995352` | 7 | pkg, debugger, `__main__` |
| `test/test_spa.py` | `D` | 2024-09-05 `8d3e41f` | 15 | pkg, sys.path, `__main__`, imports `test_cone_project` as a top-level module |

Auxiliary:

| Path | Trust | Last Commit | Commits | Signals |
|---|---|---:|---:|---|
| `test/__init__.py` | `D` | 2024-07-29 `abdd80c` | 3 | empty package marker |

## Known Runtime Blockers

- [src/irrevolutions/practice/traction-AT2_cone.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/src/irrevolutions/practice/traction-AT2_cone.py) imports `algorithms.so_merged`, which does not exist in this tree.
- [test/test_spa.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/test/test_spa.py) imports `test_cone_project` as a top-level module instead of using a package-relative import.
- Current environment check: `pytest` works, but DOLFINx-facing tests do not collect here because `dolfinx` and `ufl` are missing.

## Branch Activity Summary

### Reading The Branch Table

- `Recent 365d` is the number of commits reachable from that branch made in the last year.
- `Unique vs main` is the count of commits reachable from the branch but not from `origin/main`.
- High `unique` with low recent activity usually means dormant experimental work that was never merged.

### Local Branches

| Branch | Last Commit | Recent 365d | Unique vs `main` | Total Commits | Head |
|---|---:|---:|---:|---:|---|
| `andres-flow` | 2025-06-30 | 34 | 26 | 1083 | `db65b0d` minor logging and second order strategy |
| `andres-bifurcation-tuning` | 2025-07-03 | 20 | 12 | 1069 | `dfd279c` playing with perturbations |
| `main` | 2025-06-19 | 8 | 0 | 1057 | `c26f1d8` Fix logger reference |
| `andres-new-models` | 2025-05-27 | 3 | 0 | 1052 | `21824f4` minor |
| `andres-plates` | 2024-03-09 | 0 | 25 | 648 | `ce70e32` initial tests |
| `andres-doc` | 2024-09-24 | 0 | 23 | 900 | `6c2c93c` documentation configuration |
| `andres-discrete` | 2024-06-28 | 0 | 17 | 458 | `9aa1b0a` minor |
| `andres-conerecipe` | 2025-02-24 | 0 | 4 | 706 | `4048488` flipping  initial guess if negative |
| `andres-0.9` | 2024-11-25 | 0 | 3 | 956 | `e57dff0` minor formatting |
| `container-recipe` | 2024-03-26 | 0 | 1 | 738 | `4284715` minor |
| `jorgensd-dokken/0.8.0` | 2024-10-07 | 0 | 1 | 918 | `7d49124` merging jdokken 0.8 pr |
| `andres-aniso` | 2024-05-16 | 0 | 0 | 771 | `7c06456` Merge pull request #51 from kumiori/andres-linesearch |
| `andres-joss-review` | 2024-10-08 | 0 | 0 | 923 | `105af20` Merge branch 'main' into andres-joss-review |
| `andres-linesearch` | 2024-05-16 | 0 | 0 | 770 | `48bb5eb` custom variational iterator |
| `andres-paper` | 2024-12-05 | 0 | 0 | 941 | `ee55b98` minor, dictionary representation |
| `andres-practice` | 2024-12-06 | 0 | 0 | 1001 | `7820a15` fix typo |
| `andres-rev-patch` | 2024-03-26 | 0 | 0 | 733 | `2c65848` minor |
| `andres-update-0.7.2` | 2024-02-09 | 0 | 0 | 618 | `484cadb` minor |
| `pr-branch` | 2024-10-08 | 0 | 0 | 929 | `10d4f3c` Update to lab nightly |

### Remote Branches

| Branch | Last Commit | Recent 365d | Unique vs `origin/main` | Total Commits | Head |
|---|---:|---:|---:|---:|---|
| `origin/andres-flow` | 2025-06-24 | 29 | 21 | 1078 | `cd365af` minor enhancements |
| `origin/main` | 2025-06-19 | 8 | 0 | 1057 | `c26f1d8` Fix logger reference |
| `origin/andres-new-models` | 2025-05-27 | 3 | 0 | 1052 | `21824f4` minor |
| `origin/andres-notch` | 2023-03-20 | 0 | 51 | 283 | `c0d7964` pacman for potential computation |
| `origin/Assala_T` | 2022-03-03 | 0 | 35 | 207 | `25ed793` moving to standalone script |
| `origin/philipp-phasefield` | 2022-08-23 | 0 | 26 | 192 | `3b476fd` committing last changes |
| `origin/andres-plates` | 2024-03-09 | 0 | 25 | 648 | `ce70e32` initial tests |
| `origin/Wissam-meshes` | 2022-02-28 | 0 | 23 | 195 | `7cefd23` Merge branch 'Wissam-meshes' of github.com:kumiori/mec647 into Wissam-meshes |
| `origin/andres-doc` | 2024-09-24 | 0 | 23 | 900 | `6c2c93c` documentation configuration |
| `origin/andres-texnote` | 2023-08-03 | 0 | 19 | 324 | `73b1da6` expressions |
| `origin/francoisfernet---mesh` | 2022-02-24 | 0 | 19 | 129 | `49f42f4` Créé avec Colaboratory |
| `origin/andres-10.1016/j.cma.2018.03.012` | 2022-02-19 | 0 | 17 | 176 | `18fcc64` starting to implement 10.1016/j.cma.2018.03.012 |
| `origin/andres-discrete` | 2024-06-28 | 0 | 17 | 458 | `9aa1b0a` minor |
| `origin/philipp-cast3m` | 2022-02-23 | 0 | 16 | 112 | `40a888b` Begin some docstring |
| `origin/igor-branche-2` | 2022-03-03 | 0 | 13 | 178 | `2279678` fixing bcs |
| `origin/philipp-epMesh` | 2022-03-03 | 0 | 13 | 109 | `d2fd2fb` factoring en passant |
| `origin/andres-viscous` | 2023-10-25 | 0 | 12 | 539 | `bf198bb` check needed on newton convergence |
| `origin/seb` | 2025-03-19 | 0 | 12 | 280 | `eb19766` Update .gitignore |
| `origin/françois` | 2022-03-01 | 0 | 11 | 122 | `bac470e` Créé avec Colaboratory |
| `origin/andres-ic3cr∀k5` | 2022-11-23 | 0 | 9 | 209 | `444eeb1` minor |
| `origin/philipp-gmshError` | 2022-02-03 | 0 | 8 | 104 | `3aa63a1` Plot problem |
| `origin/francoisfernet2` | 2022-02-05 | 0 | 4 | 115 | `012a22f` Update mec647_Elast_2.ipynb |
| `origin/igor_PhaseField` | 2022-03-01 | 0 | 4 | 176 | `ee89ecd` final commit |
| `origin/andres-conerecipe` | 2024-03-22 | 0 | 3 | 705 | `f894446` merge main after rev-patch |
| `origin/assalaTrabelsi-meshes-1` | 2022-02-03 | 0 | 3 | 113 | `8b5d348` First trY FOR THE KINKING PROBLEM |
| `origin/pcesana81-patch-3` | 2024-04-11 | 0 | 3 | 758 | `59a4309` Update readme.md |
| `origin/andres-0.9` | 2024-11-25 | 0 | 2 | 955 | `0880968` fix |
| `origin/andres-fatigue` | 2022-08-31 | 0 | 2 | 189 | `131d573` basic fatigue model, first pass |
| `origin/philipp-energy` | 2022-01-31 | 0 | 2 | 98 | `9a5a603` Plot for energy |
| `origin/pierluigi-review` | 2024-08-16 | 0 | 2 | 833 | `ca0b2ff` Merge branch 'main' into pierluigi-review |
| `origin/andres-hybrid` | 2023-04-20 | 0 | 1 | 269 | `db0a779` minor |
| `origin/container-recipe` | 2024-03-26 | 0 | 1 | 738 | `4284715` minor |
| `origin/kumiori-workflow` | 2023-10-01 | 0 | 1 | 457 | `b2e31e6` Create pylint.yml |
| `origin/mert-friday` | 2023-10-26 | 0 | 1 | 448 | `6afc5e6` Save changes |
| `origin/mert-thursday_v0` | 2023-10-26 | 0 | 1 | 448 | `6afc5e6` Save changes |
| `origin/pcesana81-patch-2` | 2024-04-11 | 0 | 1 | 743 | `bac0043` Create readme.md |
| `origin/2020` | 2021-02-15 | 0 | 0 | 66 | `5d3d44b` Créé avec Colaboratory |
| `origin/andres-elasticity` | 2022-01-26 | 0 | 0 | 95 | `54636ce` ignore and utils |
| `origin/andres-joss-review` | 2024-10-08 | 0 | 0 | 923 | `105af20` Merge branch 'main' into andres-joss-review |
| `origin/andres-linesearch` | 2024-05-16 | 0 | 0 | 770 | `48bb5eb` custom variational iterator |
| `origin/andres-meshes` | 2022-01-26 | 0 | 0 | 98 | `284e586` V notch and tdcb scripts |
| `origin/andres-paper` | 2024-12-05 | 0 | 0 | 941 | `ee55b98` minor, dictionary representation |
| `origin/andres-parametric` | 2023-10-19 | 0 | 0 | 519 | `d1148ee` Merge branch 'main' into andres-parametric |
| `origin/andres-practice` | 2024-12-06 | 0 | 0 | 1001 | `7820a15` fix typo |
| `origin/andres-rev-patch` | 2024-03-26 | 0 | 0 | 733 | `2c65848` minor |
| `origin/andres-update-0.7.2` | 2024-02-09 | 0 | 0 | 618 | `484cadb` minor |
| `origin/ata-m00n` | 2022-08-31 | 0 | 0 | 189 | `e88abe1` welcome Ata |
| `origin/mert-thursday` | 2023-10-23 | 0 | 0 | 527 | `0e14940` Merge pull request #30 from kumiori/andres-conerecipe |
| `origin/pcesana81-patch-1` | 2024-03-29 | 0 | 0 | 741 | `f89fe7a` Update README.md |
| `origin/xinyuan_fenics2019` | 2023-08-06 | 0 | 0 | 283 | `9e8e460` Merge branch 'main' into xinyuan_fenics2019 |
| `origin/xinyuan_stable` | 2023-08-07 | 0 | 0 | 412 | `d567f8d` Merge pull request #11 from kumiori/andres-conerecipe |

## Branch Reading Notes

- The current active development arc is second-order/stability work: `andres-flow`, `andres-bifurcation-tuning`, then `main`.
- `andres-new-models` was active recently but appears fully merged or at least not divergent from `main`.
- `andres-doc`, `andres-discrete`, and `andres-plates` still have substantial unique history but no recent movement.
- Several remote-only branches with high uniqueness are historical experiments rather than active lines of development, especially `origin/andres-notch`, `origin/Assala_T`, and `origin/philipp-phasefield`.
