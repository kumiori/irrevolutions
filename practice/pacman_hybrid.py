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
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation as AM, HybridFractureSolver

from meshes.pacman import mesh_pacman
from utils.lib import _local_notch_asymptotic

from utils.viz import plot_mesh, plot_vector, plot_scalar
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
prefix = os.path.join(outdir, "pacman")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

def pacman_hybrid(nest):
    # Parameters
    Lx = 1.0
    Ly = 0.1
    # tdim = 2
    # _ell = 0.3
    _nel = 30

    with open(f"{prefix}/parameters.yaml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

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

    gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    if comm.rank == 0:
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

    state = {"u": u, "alpha": alpha}

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

    # Bounds for Newton solver

    u_lb = Function(V_u, name="displacement lower bound")
    u_ub = Function(V_u, name="displacement upper bound")
    alpha_lb = Function(V_alpha, name="damage lower bound")
    alpha_ub = Function(V_alpha, name="damage upper bound")
    set_vector_to_constant(u_lb.vector, PETSc.NINFINITY)
    set_vector_to_constant(u_ub.vector, PETSc.PINFINITY)
    set_vector_to_constant(alpha_lb.vector, 0)
    set_vector_to_constant(alpha_ub.vector, 1)

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    model = Brittle(parameters["model"])

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    parameters.get("model")["k_res"] = 1e-04
    parameters.get("solvers").get("damage_elasticity")["alpha_tol"] = 1e-03
    parameters.get("solvers").get("damage")["type"] = "SNES"

    Eu = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
    Ealpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha))

    F = [Eu, Ealpha]
    z = [u, alpha]

    hybrid = HybridFractureSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])

    # loads = [0.1, 1.0, 1.1]
    # loads = np.linspace(0.3, 1., 10)

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    snes = hybrid.newton.snes

    lb = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)
    ub = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)
    functions_to_vec([u_lb, alpha_lb], lb)
    functions_to_vec([u_ub, alpha_ub], ub)

    data = []

    for i_t, t in enumerate(loads):

        uD.interpolate(lambda x: _local_notch_asymptotic(
            x,
            ω=np.deg2rad(_omega / 2.),
            t=t,
            par=parameters["material"]
        ))

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.info(f"-- Solving for t = {t:3.2f} --")
        hybrid.solve()

        # compute the rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        # logging.info(f"alpha vector norm: {alpha.vector.norm()}")
        # logging.info(f"alpha lb norm: {alpha_lb.vector.norm()}")
        # logging.info(f"alphadot norm: {alphadot.vector.norm()}")
        # logging.info(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")

        # compute rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        rate_12_norm = hybrid.scaled_rate_norm(alphadot, parameters)
        rate_12_norm_unscaled = hybrid.unscaled_rate_norm(alphadot)


        fracture_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )

        datai = {
            "it": i_t,
            "AM_F_alpha_H1": hybrid.data["error_alpha_H1"][-1],
            "AM_Fnorm": hybrid.data["error_residual_F"][-1],
            "NE_Fnorm": hybrid.newton.snes.getFunctionNorm(),
            "load": t,
            "fracture_energy": fracture_energy,
            "elastic_energy": elastic_energy,
            "total_energy": elastic_energy+fracture_energy,
            "solver_data": hybrid.data,
            "rate_12_norm": rate_12_norm,
            "rate_12_norm_unscaled": rate_12_norm_unscaled
            # "eigs" : stability.data["eigs"],
            # "stable" : stability.data["stable"],
            # "F" : _F
        }

        data["it"].append(datai["it"])
        data["AM_F_alpha_H1"].append(datai["AM_F_alpha_H1"])
        data["AM_Fnorm"].append(datai["AM_Fnorm"])
        data["NE_Fnorm"].append(datai["NE_Fnorm"])
        data["load"].append(datai["load"])
        data["fracture_energy"].append(datai["fracture_energy"])
        data["elastic_energy"].append(datai["elastic_energy"])
        data["total_energy"].append(datai["total_energy"])
        data["solver_data"].append(datai["solver_data"])
        data["rate_12_norm"].append(datai["rate_12_norm"])
        data["rate_12_norm_unscaled"].append(datai["rate_12_norm_unscaled"])

        # logging.info(f"getConvergedReason() {newton.snes.getConvergedReason()}")
        # logging.info(f"getFunctionNorm() {newton.snes.getFunctionNorm():.5e}")
        try:
            check_snes_convergence(hybrid.newton.snes)
        except ConvergenceError:
            logging.info("not converged")

        # assert newton.snes.getConvergedReason() > 0


        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(data, stream=a_file, allow_unicode=True)
            a_file.close()

        ColorPrint.print_info(
            f"NEWTON - Iterations: {hybrid.newton.snes.getIterationNumber()+1:3d},\
            Fnorm: {hybrid.newton.snes.getFunctionNorm():3.4e},\
            alpha_max: {alpha.vector.max()[1]:3.4e}"
        )

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True
        plotter = pyvista.Plotter(
            title="SNES Block Restricted",
            window_size=[1600, 600],
            shape=(1, 2),
        )
        _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
        _plt = plot_vector(u, plotter, subplot=(0, 1))
        if comm.rank == 0:
            Path("output").mkdir(parents=True, exist_ok=True)
        _plt.screenshot(f"{prefix}/pacman_hybrid-{comm.size}-{i_t}.png")
        _plt.close()

    print(data)


if __name__ == "__main__":
    pacman_hybrid(nest=False)
