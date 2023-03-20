import sys
from pathlib import Path
import os
from pyvista.utilities import xvfb
import pyvista
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    locate_dofs_topological,
    set_bc,
)
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import locate_entities_boundary, CellType, create_rectangle
from dolfinx.fem import locate_dofs_topological
import yaml
import dolfinx.plot


sys.path.append("../")
from utils import ColorPrint, set_vector_to_constant
from models import ElasticityModel
from algorithms.am import AlternateMinimisation as AM, HybridFractureSolver

from meshes.pacman import mesh_embedded_pacman, mesh_cut_pacman
from utils.lib import _local_notch_asymptotic
from solvers import SNESSolver as ElasticitySolver

from utils.viz import plot_mesh, plot_vector
from mpi4py import MPI
import json
from petsc4py import PETSc
from solvers.function import functions_to_vec
from dolfinx.fem import FunctionSpace
import ufl
import petsc4py
from solvers.snesblockproblem import SNESBlockProblem
import dolfinx
from datetime import date
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)

today = date.today()


petsc4py.init(sys.argv)


comm = MPI.COMM_WORLD
# import pdb

# import pyvista


model_rank = 0


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


outdir = "output"
prefix = os.path.join(outdir, "pacman_embedded")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

def pacman_embedded(l0 = 0.1, nest = False):
    # Parameters

    Lx = 1.0
    Ly = 0.1
    _nel = 30

    with open(f"{prefix}/parameters.yaml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)


    s0 = np.linspace(0., parameters.get("geometry").get("r"), 10)
    # Mesh

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

    parameters["geometry"]["l0"] = l0

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    gmsh_model, tdim = mesh_cut_pacman(geom_type, parameters["geometry"], tdim, eta=1.e-5)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    if comm.rank == 0:
        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.pdf")

    # Function spaces
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    # Define the state
    u = Function(V_u, name="Displacement")

    state = {"u": u}

    # Data

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

    uD.interpolate(lambda x: _local_notch_asymptotic(
        x, ω=np.deg2rad(_omega / 2.), par=parameters["material"]))

    bcs_u = [dirichletbc(value=uD, dofs=boundary_dofs_u)]

    bcs = {"bcs_u": bcs_u}

    # Bounds for Newton solver

    model = ElasticityModel(parameters["model"])

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    Eu = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))


    solver = ElasticitySolver(
        Eu,
        u,
        bcs_u,
        bounds=None,
        petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
        prefix=parameters.get("solvers").get("elasticity").get("prefix"),
    )


    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])

    # if comm.rank == 0:
    #     with open(f"{prefix}/parameters.yaml", 'w') as file:
    #         yaml.dump(parameters, file)

    data = {
        "load": [],
        "elastic_energy": [],
        # "solver_data": [],
        }


    uD.interpolate(lambda x: _local_notch_asymptotic(
        x,
        ω=np.deg2rad(_omega / 2.),
        t=1.,
        par=parameters["material"]
    ))

    for i_t, t in enumerate(range(1)):

        logging.info(f"-- Solving for t = {t:3.2f} --")
        solver.solve()

        elastic_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )

        datai = {
            "load": t,
            "elastic_energy": elastic_energy,
            # "solver_data": solver.data,
        }

        data["load"].append(datai["load"])
        data["elastic_energy"].append(datai["elastic_energy"])
        # data["solver_data"].append(datai["solver_data"])


        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(data, a_file)
            a_file.close()

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True
        plotter = pyvista.Plotter(
            title="SNES Block Restricted",
            window_size=[800, 600],
            shape=(1, 1),
        )
        _plt = plot_vector(u, plotter)
        if comm.rank == 0:
            Path("output").mkdir(parents=True, exist_ok=True)
        _plt.screenshot(f"{prefix}/pacman_embedded-{comm.size}-l0={l0}.png")
        _plt.close()

    print(data)
    return (l0, elastic_energy)




def compute_release(lengths):
    
    data = {
        "l0": [],
        "elastic_energy": [],
    }
    for l in lengths:
        l0, en = pacman_embedded(l)

        data.get("l0").append(l0)
        data.get("elastic_energy").append(en)

    return data


if __name__ == "__main__":
    # pacman_embedded(l0=.3, nest=False)
    data = compute_release(np.linspace(0.01, .8, 30))

    if comm.rank == 0:
        a_file = open(f"{prefix}/time_data.json", "w")
        json.dump(data, a_file)
        a_file.close()

    import pandas as pd
    df = pd.DataFrame(data)
    print(df)
