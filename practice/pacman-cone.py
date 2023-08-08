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
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation, HybridFractureSolver
from algorithms.so import StabilitySolver, ConeSolver
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2
from meshes.pacman import mesh_pacman
from utils.viz import plot_mesh, plot_vector, plot_scalar
from utils.lib import _local_notch_asymptotic


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

comm = MPI.COMM_WORLD

outdir = "output"
prefix = os.path.join(outdir, "pacman-cone")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

def pacman_cone(nest):
    Lx = 1.0
    Ly = 0.1
    _nel = 30

    with open(f"{prefix}/parameters.yaml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
        pretty_parameters = json.dumps(parameters, indent=2)
        print(pretty_parameters)


    # Get mesh parameters
    _r = parameters["geometry"]["r"]
    _omega = parameters["geometry"]["omega"]
    tdim = parameters["geometry"]["geometric_dimension"]
    _nameExp = parameters["geometry"]["geom_type"]
    _nameExp = 'pacman'
    ell_ = parameters["model"]["ell"]
    lc = ell_ / 1.

    parameters["geometry"]["lc"] = lc

    parameters["loading"]["min"] = 0.
    parameters["loading"]["max"] = .5
    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]
    # Mesh on node model_rank and then distribute
    model_rank = 0

    gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)


    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"{prefix}/mesh.png")


    # Function spaces
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    # Define the state
    u = Function(V_u, name="Displacement")
    alpha = Function(V_alpha, name="Damage")
    alphadot = Function(V_alpha, name="Damage rate")

    # upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")


    # Pack state
    state = {"u": u, "alpha": alpha}


    uD = Function(V_u, name="Asymptotic Notch Displacement")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    # Set Bcs Function

    ext_bd_facets = locate_entities_boundary(
        mesh, dim=1, marker=lambda x: np.isclose(x[0]**2. + x[1]**2. - _r**2, 0., atol=1.e-4)
    )

    boundary_dofs_u = locate_dofs_topological(
        V_u, mesh.topology.dim - 1, ext_bd_facets)
    boundary_dofs_alpha = locate_dofs_topological(
        V_alpha, mesh.topology.dim - 1, ext_bd_facets)

    uD.interpolate(lambda x: _local_notch_asymptotic(
        x, ω=np.deg2rad(_omega / 2.), par=parameters["material"]))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)

    bcs_u = [dirichletbc(value=uD, dofs=boundary_dofs_u)]

    bcs_alpha = [
        dirichletbc(
            np.array(0, dtype=PETSc.ScalarType),
            boundary_dofs_alpha,
            V_alpha,
        )
    ]

    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    bcs_z = bcs_u + bcs_alpha

    # Mechanical model

    model = Brittle(parameters["model"])

    # Energy functional

    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])

    # Solvers

    solver = AlternateMinimisation(
        total_energy, state, bcs, parameters.get("solvers"), 
        bounds=(alpha_lb, alpha_ub)
    )

    hybrid = HybridFractureSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    stability = StabilitySolver(
        total_energy, state, bcs, stability_parameters=parameters.get("stability")
    )

    cone = ConeSolver(
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
        "cone_data": [],
        "cone-eig": [],
        "eigs": [],
        "uniqueness": [],
        "inertia": [],
        "stable": [],
        "F": [],    
        "alphadot_norm" : [],
        "rate_12_norm" : [], 
        "unscaled_rate_12_norm" : [],
        "cone-stable": []
    }


    for i_t, t in enumerate(loads):

        uD.interpolate(lambda x: _local_notch_asymptotic(
            x,
            ω=np.deg2rad(_omega / 2.),
            t=t,
            par=parameters["material"]
        ))


if __name__ == "__main__":
    pacman_cone(nest=False)
