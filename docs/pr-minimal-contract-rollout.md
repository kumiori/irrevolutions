# PR Note: Minimal Contract Rollout

## Title

Introduce minimal experiment contract and migrate first public scripts

## Summary

This PR introduces a minimal formal contract for experiment scripts and applies
it to the first batch of validated entry points.

The contract is intentionally small:

- `ExperimentSetup`
- `EquilibriumResult`
- `StepRecord`
- `History`
- `Manifest`

The goal of this PR is not to introduce a shared runner yet. The goal is to
make the runtime objects explicit and stable before any orchestration layer is
built on top of them.

## What the contract defines

The contract separates five concerns that were previously mixed together inside
application scripts:

- `ExperimentSetup`
- `EquilibriumResult`
- `StepRecord`
- `History`
- `Manifest`

More concretely, it gives explicit names and scopes to:

- the live mathematical `state`
- the boundary and loading description `bcs`
- the admissibility structure `bounds`
- the accepted-step scientific log `History`
- the run-level provenance record `Manifest`

This matters because the code currently mixes:

- live unknowns
- solver workspaces
- history columns
- postprocessed observables
- provenance metadata

inside the same script-level objects. The contract makes those roles explicit.

## Intended scope

This contract is intentionally minimal and is aimed first at the application
family that dominates the repository today:

- traction and bar evolution scripts
- first-order equilibrium followed by bifurcation or cone stability analysis
- output pipelines that currently write XDMF, JSON, tables, and plots

The first migrated scripts were chosen because they span that scope clearly:

- [test/test_1d.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/test/test_1d.py)
- [test/contract/test_1d.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/test/contract/test_1d.py)
- [demo/demo_traction.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/demo/demo_traction.py)
- [practice/traction-bar-clean.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-bar-clean.py)
- [practice/traction-cone.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-cone.py)

This does not claim that every historical playground script already fits the
contract. It claims that the contract is now well defined for the intended,
high-signal application family and validated on representative scripts from
that family.

## Why a contract matters

### Transparency

The contract makes the runtime handoff explicit.

Before:

- second-order analysis implicitly relied on mutable script state and side effects

After:

- first-order accepted equilibria are named as `EquilibriumResult`
- accepted load steps are named as `StepRecord`
- run-level provenance is named separately as `Manifest`

This makes the scientific workflow easier to read, reason about, and review.

### Reproducibility

The contract separates:

- step-level observables in `History`
- run-level metadata in `Manifest`

That separation is important because reproducibility is not only about keeping
parameter files. It is about making it unambiguous which data describe:

- the current state of the model
- the accepted scientific outputs at each step
- the provenance of the run that produced them

This reduces ambiguity when comparing runs, exporting results, or revisiting
older branches.

### Scalability

The contract is a prerequisite for scaling the codebase in two directions:

- more scripts sharing the same solve choreography
- more models carrying additional internal variables beyond `u` and `alpha`

Without a contract, a future runner would simply freeze the current accidental
script conventions. With a contract, the runner can remain thin and explicit.

This is also the point where adding new model fields such as plastic or phase
variables becomes a controlled extension of the interface instead of another
round of ad hoc script branching.

## What changed

### New contract layer

- added [contracts.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/src/irrevolutions/contracts.py)
- added [minimal-contract.md](/Users/kumiori3/Documents/WIP/Nature/code_mec647/docs/minimal-contract.md)

This layer formalizes:

- live model `state`
- structured `bcs`
- named `bounds`
- accepted-step `History`
- run-level `Manifest`

It also includes small adapter helpers so existing solver interfaces can be
used without introducing the runner yet.

The contract is therefore not only a refactor of code shape. It is a
formalization of the scientific data flow through the application scripts.

### First migrated tests

- adapted [test/test_1d.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/test/test_1d.py)
- added [test/contract/test_1d.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/test/contract/test_1d.py)

The contract test gives a minimal reference implementation of the first-order
to second-order handoff using the explicit contract objects.

### First migrated applications

- adapted [demo/demo_bifurcation.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/demo/demo_bifurcation.py)
- adapted [demo/demo_elasticity.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/demo/demo_elasticity.py)
- adapted [demo/demo_traction.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/demo/demo_traction.py)
- adapted [practice/traction-AT1_cone.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-AT1_cone.py)
- adapted [practice/traction-ATJJ.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-ATJJ.py)
- adapted [practice/traction-bar-clean.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-bar-clean.py)
- adapted [practice/traction-cone.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/practice/traction-cone.py)

These scripts now use the contract objects while preserving their current
numerical flow and output structure.

This is the key claim of the rollout:

- the contract is explicit
- it is implemented
- and it is already usable in representative applications without changing the
  scientific problem being solved

### Validation tooling

- added [run_demo_matrix.py](/Users/kumiori3/Documents/WIP/Nature/code_mec647/tools/run_demo_matrix.py)

This mirrors the practice script matrix and gives a single entrypoint for
checking demo health.

## Validation

Validated in the DOLFINx 0.9 container:

- `pytest -q test/test_1d.py`
- `pytest -q test/contract/test_1d.py`
- `python3 tools/run_demo_matrix.py`

Current demo matrix result:

- 7 demos passed
- 1 demo failed: `demo_biharmonic`

Important:

- `demo_biharmonic` is not failing because of the contract rollout
- it currently depends on parameters/assets that live in another project
- this PR treats it as an external-input issue, not as a regression of the
  migrated code path

## Why this ordering

The contract is introduced before any shared runner because the runner should
encode an explicit interface, not crystallize accidental conventions from the
legacy scripts.

This PR therefore focuses on:

1. naming the runtime objects
2. validating them in one test and a few high-signal scripts
3. preserving existing behavior
