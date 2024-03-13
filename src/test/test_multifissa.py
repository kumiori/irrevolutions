import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO)
from datetime import date

today = date.today()
sys.path.append("../")

import dolfinx
from solvers.snesblockproblem import SNESBlockProblem
import petsc4py
import ufl
from dolfinx.fem import FunctionSpace
from solvers.function import functions_to_vec
from petsc4py import PETSc
import json
from utils.plots import plot_energies

petsc4py.init(sys.argv)

from mpi4py import MPI
from utils.viz import plot_mesh, plot_vector, plot_scalar

comm = MPI.COMM_WORLD
# import pdb
import dolfinx.plot
import matplotlib.pyplot as plt
# import pyvista
import yaml
from algorithms.am import AlternateMinimisation as AM, HybridSolver
from models import BrittleMembraneOverElasticFoundation as ThinFilm
from utils import ColorPrint, set_vector_to_constant
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary, CellType, create_rectangle

import pyvista
from pyvista.utilities import xvfb


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


import os
from pathlib import Path

outdir = "output"
prefix = os.path.join(outdir, "multifissa")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

def test_multifissa(nest):
    Lx = 1.
    Ly = 0.05
    _meshsize = Lx / 100

    try:
        with open(f"{prefix}/parameters.yaml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)
            Lx = parameters.get("geometry").get("Lx")
            Ly = parameters.get("geometry").get("Ly")
            _meshsize = parameters.get("model").get("ell") / parameters.get("geometry").get("elltomesh")

    except IOError:
        logging.info('No parameters found, creating new.')
        if comm.rank == 0:
            with open(f"{prefix}/parameters.yaml", 'w') as file:
                yaml.dump({}, file)

    _nel = int(1./_meshsize)

    logging.info(f'Mesh reslution 1/nel (Lx + Ly/Lx) {1/_nel*(Lx + Ly/Lx)}')
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [int(_nel * Lx), int(_nel * Ly)],
        CellType.triangle,
    )
    coord_dofs = mesh.geometry.dofmap
    logging.info(f"# dofs: {3* len(coord_dofs)}")

    if comm.rank == 0:
        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")



    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

    # Define the state
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage Rate")

    state = {"u": u, "alpha": alpha}

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    # ds = ufl.Measure("ds", domain=mesh)

    zero_u = dolfinx.fem.Function(V_u, name="   Boundary Displacement")
    set_vector_to_constant(zero_u.vector, 0.0)

    zero_alpha = dolfinx.fem.Function(V_alpha, name="Damage Boundary Field")
    set_vector_to_constant(zero_alpha.vector, 0.0)

    u_lb = dolfinx.fem.Function(V_u, name="displacement lower bound")
    u_ub = dolfinx.fem.Function(V_u, name="displacement upper bound")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="damage lower bound")
    alpha_ub = dolfinx.fem.Function(V_alpha, name="damage upper bound")
    set_vector_to_constant(u_lb.vector, PETSc.NINFINITY)
    set_vector_to_constant(u_ub.vector, PETSc.PINFINITY)
    set_vector_to_constant(alpha_lb.vector, 0)
    set_vector_to_constant(alpha_ub.vector, 1)

    u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")
    u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], Lx)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_dofs_1 = locate_dofs_topological(V_u, mesh.topology.dim - 1, left_facets)
    left_dofs_2 = locate_dofs_topological(V_alpha, mesh.topology.dim - 1, left_facets)

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_dofs_1 = locate_dofs_topological(V_u, mesh.topology.dim - 1, right_facets)
    right_dofs_2 = locate_dofs_topological(V_alpha, mesh.topology.dim - 1, right_facets)

    bcs_u = [
        # dolfinx.fem.dirichletbc(zero_u, left_dofs_1),
        # dolfinx.fem.dirichletbc(u_, right_dofs_1),
    ]
    bcs_alpha = [
        dolfinx.fem.dirichletbc(zero_alpha, left_dofs_2),
        dolfinx.fem.dirichletbc(zero_alpha, right_dofs_2),
    ]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    _t = dolfinx.fem.Constant(mesh, 1.)

    model = ThinFilm(parameters["model"], eps_0= _t * ufl.Identity(2))

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

    hybrid = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])


    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)


    snes = hybrid.newton.snes

    lb = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)
    ub = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)

    functions_to_vec([u_lb, alpha_lb], lb)
    functions_to_vec([u_ub, alpha_ub], ub)

    # data = []

    data = {
        "it": [],
        "AM_F_alpha_H1": [],
        "AM_Fnorm": [],
        "NE_Fnorm": [],
        "load": [],
        "fracture_energy": [],
        "elastic_energy": [],
        "total_energy": [],
        "solver_data": [],
        "rate_12_norm": [],
        "rate_12_norm_unscaled": []
        }

    for i_t, t in enumerate(loads):

        _t.value = t
        
        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.info(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")
        logging.info(f"-- Solving for t = {t:3.2f} --")

        hybrid.solve()
        
        fracture_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )

        # compute rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        rate_12_norm = hybrid.scaled_rate_norm(alphadot, parameters)
        rate_12_norm_unscaled = hybrid.unscaled_rate_norm(alphadot)

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
        # "eigs" : stability.data["eigs"],
        # "stable" : stability.data["stable"],
        # "F" : _F

        # data.append(datai)

        logging.critical(f"getConvergedReason() {hybrid.newton.snes.getConvergedReason()}")
        logging.critical(f"getFunctionNorm() {hybrid.newton.snes.getFunctionNorm():.5e}")
        
        try:
            check_snes_convergence(hybrid.newton.snes)
        except ConvergenceError:
            logging.info("not converged")

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(data, a_file)
            a_file.close()

        ColorPrint.print_info(
            f"NEWTON - Iterations: {hybrid.newton.snes.getIterationNumber()+1:3d},\
            Fnorm: {hybrid.newton.snes.getFunctionNorm():3.4e},\
            alpha_max: {alpha.vector.max()[1]:3.4e}"
        )


        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True
        plotter = pyvista.Plotter(
            title="Multifissuration",
            window_size=[1600, 600],
            shape=(1, 2),
        )
        _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
        _plt = plot_vector(u, plotter, subplot=(0, 1))
        if comm.rank == 0:
            Path("output").mkdir(parents=True, exist_ok=True)
        _plt.screenshot(f"{prefix}/test_multifissa-{comm.size}-{i_t}.png")
        _plt.close()

    print(data)

    if comm.rank == 0:
        plot_energies(data, file=f"{prefix}/energies.pdf")

if __name__ == "__main__":
    test_multifissa(nest=False)
