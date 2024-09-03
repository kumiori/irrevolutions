#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import pandas as pd
import yaml
from dolfinx.common import list_timings

#
from mpi4py import MPI
from petsc4py import PETSc

sys.path.append("../")

logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------
class ConvergenceError(Exception):
    """Error raised when a solver fails to converge"""


def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r) for r in dir(reasons) if not r.startswith("_")]
    )


SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())
KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())


def check_snes_convergence(snes):
    r = snes.getConvergedReason()
    try:
        reason = SNESReasons[r]
        inner = False
        logging.info(f"snes converged with reason {r}: {reason}")
    except KeyError:
        r = snes.getKSP().getConvergedReason()
        try:
            inner = True
            reason = KSPReasons[r]
        except KeyError:
            reason = "unknown reason (petsc4py enum incomplete?), try with -snes_converged_reason and -ksp_converged_reason"
    if r < 0:
        if inner:
            msg = (
                "Inner linear solve failed to converge after %d iterations with reason: %s"
                % (snes.getKSP().getIterationNumber(), reason)
            )
        else:
            msg = reason
        raise ConvergenceError(
            r"""Nonlinear solve failed to converge after %d nonlinear iterations.
                Reason:
                %s"""
            % (snes.getIterationNumber(), msg)
        )


# ------------------------------------------------------------------


comm = MPI.COMM_WORLD


outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "multiaxial-disc")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)


def multiaxial_disc(nest):
    """Testing nucleation for for a multiaxial disc,
    thanks to: Camilla Zolesi"""

    # parameters: INPUT

    with open("../test/parameters.yml") as f:
        yaml.load(f, Loader=yaml.FullLoader)

    # history_data: OUTPUT

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
        "cone_data": [],
        "cone_eig": [],
        "eigs": [],
        "uniqueness": [],
        "inertia": [],
        "stable": [],
        "alphadot_norm": [],
        "rate_12_norm": [],
        "unscaled_rate_12_norm": [],
        "cone-stable": [],
    }

    # generate mesh

    # functional space

    # boundary conditions

    # energy (model) <--------------
    # total_energy = ThinFilmModel(parameters...)

    # solvers

    # timestepping

    # postprocessing

    return history_data


if __name__ == "__main__":
    history_data = multiaxial_disc(nest=False)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    df = pd.DataFrame(history_data)
    print(df.drop(["solver_data", "cone_data"], axis=1))
