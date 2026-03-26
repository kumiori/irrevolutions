# Minimal Contract

This document defines the minimal formal contract for experiment scripts in
`irrevolutions`. The goal is to make the handoff between first-order solve,
second-order analysis, history logging, and provenance explicit before a
shared runner is introduced.

## Objects

The current vocabulary is:

- `ExperimentSetup`
- `EquilibriumResult`
- `StepRecord`
- `History`
- `Manifest`

The first three are runtime objects. `History` is the append-only collection of
accepted load steps. `Manifest` is the run-level provenance record.

## State

`state` is the live current mathematical state of the model at the accepted
load step.

It contains:

- primal unknowns
- constitutive or internal variables needed to define energy, admissibility, or evolution

It does not contain:

- postprocessed observables
- history arrays
- solver workspaces
- purely diagnostic vectors

Current required fields for the damage/fracture engine:

- `state["u"]`
- `state["alpha"]`

Contract:

```python
state: Mapping[str, fem.Function]
```

Rules:

- each value is a live `dolfinx.fem.Function`
- objects are mutated in place
- field names are part of the public contract inside a model family

## BCs

`bcs` is the boundary and loading specification for the current step.

Contract:

```python
bcs = {
    "u": {
        "dirichlet": [...],
        "loading": {
            "type": "displacement_control",
            "parameter": t,
            "component": 1,
            "region": "top",
        },
    },
    "alpha": {
        "dirichlet": [],
        "loading": None,
    },
}
```

Rules:

- keys refer to fields in `state`
- `dirichlet` is always present and is always a list
- `loading` is optional
- BCs describe imposed actions, not observables or solver counters

Current minimum required keys for the damage scripts are `u` and `alpha`.

## Bounds

`bounds` is the admissibility structure of constrained fields.

Contract:

```python
bounds = {
    "alpha": {
        "kind": "field",
        "lower": alpha_lb,
        "upper": alpha_ub,
    }
}
```

Rules:

- bounds are always attached by field name
- lower and upper bounds are explicit
- fields with no admissibility constraints are absent from `bounds`
- irreversibility is represented by mutating the relevant lower bound in place
- no anonymous `(lb, ub)` pairs should float around without a field name

The current essential case is full-field bounds on `alpha`.

## EquilibriumResult

`EquilibriumResult` is the accepted output of the first-order solve at one load
step.

It wraps:

- the accepted live `state`
- the active `bounds`
- convergence status
- top-level first-order diagnostics

It is intentionally more than `state`, so the second-order analysis consumes an
explicit result rather than an implicit bundle of mutable side effects.

## StepRecord and History

One `StepRecord` corresponds to one accepted load step. `History` is an
append-only collection of `StepRecord` objects.

Runtime canonical form:

```python
History(records=[StepRecord(...), StepRecord(...), ...])
```

Columnar export form:

```python
history_data = history.to_columns()
```

Rules:

- one new row per accepted load step
- all columns have the same length
- no provenance metadata in `history_data`
- absent information is explicit, using `None` and status flags

Recommended explicit second-order fields:

- `stability_attempted`
- `stability_converged`
- `stable`
- `lambda_stab_min`
- `bifurcation_attempted`
- `bifurcation_converged`
- `unique`
- `lambda_bif_min`

Failure encoding rule:

- if second-order analysis is requested but fails, the step still records
  attempted status, non-convergence, and `None` values for the unresolved
  scientific quantities

## Manifest

`Manifest` is the run-level provenance record. It is separate from
`History`.

Suggested contents:

- resolved parameters
- git commit / dirty state
- mesh metadata
- function space metadata
- solver options
- run identifier
- timestamps
- environment details

Rule:

- `Manifest` stores reproducibility metadata, not step-level observables

## Current adapters

The current solver APIs still use older conventions in several places:

- BCs as `{"bcs_u": [...], "bcs_alpha": [...]}`
- bounds as positional `(lower, upper)` tuples for damage

The initial contract layer keeps this explicit with helper adapters:

- `normalise_bcs(...)`
- `legacy_bcs_from_contract(...)`
- `make_field_bounds(...)`
- `get_bounds_pair(...)`

This keeps the new contract explicit without forcing a runner refactor first.
