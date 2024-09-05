#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista
import ufl
import yaml
from dolfinx.common import list_timings, timing
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import locate_entities_boundary

#
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.utilities import xvfb
import basix.ufl

sys.path.append("../")
from algorithms.am import AlternateMinimisation, HybridSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.utils import ColorPrint
from meshes.pacman import mesh_pacman
from models import DamageElasticityModel as Brittle
from utils.lib import _local_notch_asymptotic
from utils.viz import plot_mesh, plot_scalar, plot_vector

logging.basicConfig(level=logging.DEBUG)

logging.logMultiprocessing = False


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


comm = MPI.COMM_WORLD


def pacman_cone(resolution=2, slug="pacman"):
    Lx = 1.0
    Ly = 0.1
    _nel = 30

    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = os.path.join(outdir, "pacman-cone")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}/parameters.yaml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
        # pretty_parameters = json.dumps(parameters, indent=2)
        # print(pretty_parameters)

    parameters["stability"]["cone"]["cone_max_it"] = 30000
    parameters["stability"]["cone"]["cone_atol"] = 1e-4
    parameters["stability"]["cone"]["scaling"] = 0.1

    # Get mesh parameters
    _r = parameters["geometry"]["r"]
    _omega = parameters["geometry"]["omega"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    _nameExp = "pacman"

    ell_ = parameters["model"]["ell"]
    lc = ell_ / resolution

    parameters["geometry"]["lc"] = lc

    parameters["loading"]["min"] = 0.35
    parameters["loading"]["max"] = 0.50
    parameters["loading"]["steps"] = 100

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]

    model_rank = 0
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    outdir = os.path.join("output", slug, signature)
    prefix = os.path.join(outdir)

    gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    # from dolfinx.mesh import refine
    # mesh.topology.create_entities(1)
    # mesh2 = refine(mesh, redistribute=True)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

        with open(f"{prefix}/parameters.yaml") as f:
            _parameters = yaml.load(f, Loader=yaml.FullLoader)

        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"{prefix}/mesh.png")

    # Function spaces
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
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
        mesh,
        dim=1,
        marker=lambda x: np.isclose(
            x[0] ** 2.0 + x[1] ** 2.0 - _r**2, 0.0, atol=1.0e-4
        ),
    )

    boundary_dofs_u = locate_dofs_topological(V_u, mesh.topology.dim - 1, ext_bd_facets)
    boundary_dofs_alpha = locate_dofs_topological(
        V_alpha, mesh.topology.dim - 1, ext_bd_facets
    )

    uD.interpolate(
        lambda x: _local_notch_asymptotic(
            x, ω=np.deg2rad(_omega / 2.0), par=parameters["material"]
        )
    )

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

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
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    # Solvers

    solver = AlternateMinimisation(
        total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
    )

    hybrid = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    cone = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
        "solver_HY_data": [],
        "solver_KS_data": [],
        # "cone-eig": [],
        "eigs": [],
        "uniqueness": [],
        "inertia": [],
        "F": [],
        "alphadot_norm": [],
        "rate_12_norm": [],
        "unscaled_rate_12_norm": [],
        "cone-stable": [],
    }

    for i_t, t in enumerate(loads):
        uD.interpolate(
            lambda x: _local_notch_asymptotic(
                x, ω=np.deg2rad(_omega / 2.0), t=t, par=parameters["material"]
            )
        )

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        ColorPrint.print_pass(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

        ColorPrint.print_bold("   Solving first order: AM*Hybrid   ")
        ColorPrint.print_bold("===================-=============")

        hybrid.solve(alpha_lb)

        # compute the rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
        urate_12_norm = hybrid.unscaled_rate_norm(alpha)

        ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold("===================-=================")

        is_stable = bifurcation.solve(alpha_lb)
        is_elastic = bifurcation.is_elastic()
        inertia = bifurcation.get_inertia()

        ColorPrint.print_bold("   Solving second order: Cone Pb.    ")
        ColorPrint.print_bold("===================-=================")

        stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum)

        logging.critical(f"State is elastic: {is_elastic}")
        logging.critical(f"State's inertia: {inertia}")

        # Postprocess

        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        history_data["solver_data"].append(hybrid.data)
        history_data["solver_HY_data"].append(hybrid.newton_data)
        history_data["solver_KS_data"].append(cone.data)
        history_data["eigs"].append(bifurcation.data["eigs"])
        history_data["F"].append(0)
        history_data["alphadot_norm"].append(alphadot.vector.norm())
        history_data["rate_12_norm"].append(rate_12_norm)
        history_data["unscaled_rate_12_norm"].append(urate_12_norm)
        history_data["cone-stable"].append(stable)
        # history_data["cone-eig"].append(cone.data["lambda_0"])
        history_data["uniqueness"].append(_unique)
        history_data["inertia"].append(inertia)

        # Save solution

        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)
            file.write_function(alphadot, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        # Viz
        if "SINGULARITY_CONTAINER" not in os.environ:
            from utils.plots import (
                plot_AMit_load,
                plot_energies,
            )

            if comm.rank == 0:
                plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
                plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
                # plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")

            ColorPrint.print_bold("   Written timely data.    ")
            print()
            print()
            print()
            print()

            xvfb.start_xvfb(wait=0.05)
            pyvista.OFF_SCREEN = True

            plotter = pyvista.Plotter(
                title="Pacman test",
                window_size=[1600, 600],
                shape=(1, 2),
            )
            _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
            _plt = plot_vector(u, plotter, subplot=(0, 1))
            _plt.screenshot(f"{prefix}/pacman-state.png")

        # __import__('pdb').set_trace()

        # plotter = pyvista.Plotter(
        #     title="Pacman bifurcations",
        #     window_size=[1600, 600],
        #     shape=(1, 2),
        # )

    _timings = list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    performance = {
        "N": [],
        "dofs": [],
        "1stOrder-AM": [],
        "1stOrder-Hyb": [],
        "1stOrder-AM-Damage": [],
        "1stOrder-AM-Elastic": [],
        "2ndOrder-Uniqueness": [],
        "2ndOrder-Stability": [],
    }

    performance["N"].append(MPI.COMM_WORLD.size)
    performance["dofs"].append(
        sum([V.dofmap.bs * V.dofmap.index_map.size_global for V in [V_u, V_alpha]])
    )
    performance["1stOrder-AM"].append(timing("~First Order: AltMin solver"))
    performance["1stOrder-Hyb"].append(timing("~First Order: Hybrid solver"))
    performance["1stOrder-AM-Damage"].append(
        timing("~First Order: AltMin-Damage solver")
    )
    performance["1stOrder-AM-Elastic"].append(
        timing("~First Order: AltMin-Elastic solver")
    )
    performance["2ndOrder-Uniqueness"].append(timing("~Second Order: Bifurcation"))

    try:
        performance["2ndOrder-Stability"].append(timing("~Second Order: Cone Solver"))
    except Exception:
        performance["2ndOrder-Stability"].append(np.nan)

    if comm.rank == 0:
        a_file = open(f"{prefix}/performance.json", "w")
        json.dump(performance, a_file)
        a_file.close()

    return history_data, signature, prefix, performance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process evolution.")
    parser.add_argument("-r", type=int, default=3, help="resolution: ell to h ratio")
    args = parser.parse_args()

    ColorPrint.print_info(f"Resolution: {args.r}")

    history_data, signature, prefix, timings = pacman_cone(resolution=args.r)
    ColorPrint.print_bold(f"   signature {signature}    ")

    df = pd.DataFrame(history_data)
    print(df.drop(["solver_data", "solver_HY_data", "solver_KS_data"], axis=1))
