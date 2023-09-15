#!/usr/bin/env python3
import pdb
import pandas as pd
import numpy as np
from sympy import derive_by_array
import yaml
import json
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

from dolfinx.fem import locate_dofs_geometrical, dirichletbc
from dolfinx.mesh import CellType
import dolfinx.mesh
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)

import pyvista
from pyvista.utilities import xvfb
# 
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
from dolfinx.mesh import locate_entities_boundary, CellType, create_rectangle
from dolfinx.fem import locate_dofs_topological

from dolfinx.fem.petsc import (
    set_bc,
    )
from dolfinx.io import XDMFFile, gmshio
import logging
from dolfinx.common import Timer, list_timings, TimingType

sys.path.append("../")
from models import BrittleMembraneOverElasticFoundation as ThinFilm
from algorithms.am import AlternateMinimisation, HybridFractureSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2
from meshes.pacman import mesh_pacman
from utils.viz import plot_mesh, plot_vector, plot_scalar
from utils.lib import _local_notch_asymptotic
logging.basicConfig(level=logging.DEBUG)



# ------------------------------------------------------------------
class ConvergenceError(Exception):
    """Error raised when a solver fails to converge"""

def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r)
         for r in dir(reasons) if not r.startswith("_")]
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


outdir = "output"
prefix = os.path.join(outdir, "thinfilm-bar")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)


def thinfilm_bar(nest):
    """Testing nucleation of patterns"""

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
        "alphadot_norm" : [],
        "rate_12_norm" : [], 
        "unscaled_rate_12_norm" : [],
        "cone-stable": []
    }

    # parameters
    geom_type = "bar"

    with open("../data/thinfilm/parameters.yml") as f:
        # default parameters
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]

    tdim = parameters["geometry"]["geometric_dimension"]
    ell_ = parameters["model"]["ell"]
    lc = ell_ / parameters["geometry"]["mesh_size_factor"]

    import hashlib
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    outdir = "output"
    prefix = os.path.join(outdir, "thinfilm-bar", signature)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    # generate mesh
    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

    # Get mesh and meshtags
    model_rank = 0
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    # functional space
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    u = Function(V_u, name="Displacement")
    zero_u = Function(V_u, name="   Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    zero_alpha = Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

    state = {"u": u, "alpha": alpha}

    z = [u, alpha]
    # need upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)


    # boundary conditions
    bcs_u = []
    bcs_alpha = []
    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}


    # loading
    tau = Constant(mesh, np.array(0., dtype=PETSc.ScalarType))
    eps_0 = tau * ufl.as_tensor([[1., 0], [0, 0]])

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])
    # energy (model)
    model = ThinFilm(parameters["model"], eps_0=eps_0)
    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    # solvers

    equilibrium = HybridFractureSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    # timestepping
    for i_t, t in enumerate(loads):
        tau.value = t

    # postprocessing


    return history_data

if __name__ == "__main__":
    history_data = thinfilm_bar(nest=False)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    
    df = pd.DataFrame(history_data)
    print(df.drop(['solver_data', 'cone_data'], axis=1))

