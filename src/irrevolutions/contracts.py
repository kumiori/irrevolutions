from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from dolfinx import fem


CURRENT_DAMAGE_FIELD_KEYS = frozenset({"u", "alpha"})


def validate_state(
    state: Mapping[str, fem.Function],
    required_keys: frozenset[str] = CURRENT_DAMAGE_FIELD_KEYS,
) -> Mapping[str, fem.Function]:
    missing = required_keys.difference(state.keys())
    if missing:
        raise ValueError(f"State is missing required fields: {sorted(missing)}")

    for name, value in state.items():
        if not isinstance(value, fem.Function):
            raise TypeError(f"State field '{name}' must be a dolfinx.fem.Function")

    return state


def normalise_bcs(
    bcs: Mapping[str, Any],
    required_keys: frozenset[str] = CURRENT_DAMAGE_FIELD_KEYS,
) -> dict[str, dict[str, Any]]:
    normalised: dict[str, dict[str, Any]] = {}

    for key, value in bcs.items():
        if key.startswith("bcs_"):
            field_name = key.removeprefix("bcs_")
            normalised[field_name] = {"dirichlet": list(value), "loading": None}
            continue

        if isinstance(value, Mapping):
            dirichlet = value.get("dirichlet", [])
            loading = value.get("loading")
            normalised[key] = {"dirichlet": list(dirichlet), "loading": loading}
            continue

        if isinstance(value, list):
            normalised[key] = {"dirichlet": list(value), "loading": None}
            continue

        raise TypeError(
            f"Boundary conditions for '{key}' must be a mapping or a list of DirichletBCs"
        )

    missing = required_keys.difference(normalised.keys())
    if missing:
        raise ValueError(f"Boundary conditions are missing required fields: {sorted(missing)}")

    return normalised


def validate_bounds(bounds: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    for field_name, spec in bounds.items():
        if "lower" not in spec or "upper" not in spec:
            raise ValueError(
                f"Bounds for '{field_name}' must define explicit 'lower' and 'upper' entries"
            )

    return bounds


def make_field_bounds(
    lower: fem.Function,
    upper: fem.Function,
    *,
    kind: str = "field",
) -> dict[str, Any]:
    return {"kind": kind, "lower": lower, "upper": upper}


def get_bounds_pair(
    bounds: Mapping[str, Mapping[str, Any]],
    field_name: str,
) -> tuple[fem.Function, fem.Function]:
    if field_name not in bounds:
        raise KeyError(f"Bounds for field '{field_name}' are not defined")

    spec = bounds[field_name]
    return spec["lower"], spec["upper"]


def legacy_bcs_from_contract(bcs: Mapping[str, Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {
        f"bcs_{field_name}": list(spec.get("dirichlet", []))
        for field_name, spec in bcs.items()
    }


@dataclass(slots=True)
class ExperimentSetup:
    state: MutableMapping[str, fem.Function]
    bcs: Mapping[str, Mapping[str, Any]]
    bounds: Mapping[str, Mapping[str, Any]]
    parameters: Mapping[str, Any]
    energy: Any
    mesh: Any | None = None
    spaces: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_state(self.state)
        self.bcs = normalise_bcs(self.bcs)
        validate_bounds(self.bounds)


@dataclass(slots=True)
class EquilibriumResult:
    step: int
    load: float
    time: float | None
    state: Mapping[str, fem.Function]
    bounds: Mapping[str, Mapping[str, Any]]
    converged: bool
    solver_name: str
    iterations: int | None = None
    residual_norm: float | None = None
    total_energy: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_state(self.state)
        validate_bounds(self.bounds)


@dataclass(slots=True)
class StepRecord:
    step: int
    load: float
    time: float | None
    elastic_energy: float | None
    fracture_energy: float | None
    total_energy: float | None
    solver_converged: bool
    n_iterations: int | None = None
    inertia: tuple[int, int, int] | None = None
    stability_attempted: bool = False
    stability_converged: bool | None = None
    stable: bool | None = None
    lambda_stab_min: float | None = None
    bifurcation_attempted: bool = False
    bifurcation_converged: bool | None = None
    unique: bool | None = None
    lambda_bif_min: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class History:
    records: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_columns(self) -> dict[str, list[Any]]:
        columns: dict[str, list[Any]] = {
            "step": [],
            "load": [],
            "time": [],
            "elastic_energy": [],
            "fracture_energy": [],
            "total_energy": [],
            "solver_converged": [],
            "n_iterations": [],
            "inertia": [],
            "stability_attempted": [],
            "stability_converged": [],
            "stable": [],
            "lambda_stab_min": [],
            "bifurcation_attempted": [],
            "bifurcation_converged": [],
            "unique": [],
            "lambda_bif_min": [],
        }

        extra_keys = sorted({key for record in self.records for key in record.extra})
        for key in extra_keys:
            columns[key] = []

        for record in self.records:
            columns["step"].append(record.step)
            columns["load"].append(record.load)
            columns["time"].append(record.time)
            columns["elastic_energy"].append(record.elastic_energy)
            columns["fracture_energy"].append(record.fracture_energy)
            columns["total_energy"].append(record.total_energy)
            columns["solver_converged"].append(record.solver_converged)
            columns["n_iterations"].append(record.n_iterations)
            columns["inertia"].append(record.inertia)
            columns["stability_attempted"].append(record.stability_attempted)
            columns["stability_converged"].append(record.stability_converged)
            columns["stable"].append(record.stable)
            columns["lambda_stab_min"].append(record.lambda_stab_min)
            columns["bifurcation_attempted"].append(record.bifurcation_attempted)
            columns["bifurcation_converged"].append(record.bifurcation_converged)
            columns["unique"].append(record.unique)
            columns["lambda_bif_min"].append(record.lambda_bif_min)

            for key in extra_keys:
                columns[key].append(record.extra.get(key))

        return columns


@dataclass(slots=True)
class Manifest:
    parameters: Mapping[str, Any]
    run_id: str | None = None
    git_commit: str | None = None
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    mesh: Mapping[str, Any] = field(default_factory=dict)
    spaces: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    timestamps: Mapping[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
