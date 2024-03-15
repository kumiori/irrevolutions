#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (assemble_scalar, form)
from dolfinx.io import XDMFFile
from mpi4py import MPI
comm = MPI.COMM_WORLD

from petsc4py import PETSc

sys.path.append("../")
from irrevolutions.utils import ColorPrint



"""Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""


logging.getLogger().setLevel(logging.CRITICAL)


class ConstrainedEvolution:
    """A Problem, solved.
        We consider the following problem:
        ...
    """

    # has: context and solver(s)


def setup(custom_parameters):
    """docstring for setup"""
    
    from dolfinx.fem.FunctionSpace import Function
    import ufl

    with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["cone"] = ""
    # parameters["cone"]["atol"] = 1e-7

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = '1D'
    parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1
    parameters["model"]["k_res"] = 1e-4
    parameters["model"]["k"] = 3
    parameters["model"]["N"] = 3
    parameters["loading"]["min"] = .5
    parameters["loading"]["max"] = 2
    parameters["loading"]["steps"] = 50
    parameters["geometry"]["geom_type"] = "discrete-damageable"

    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    parameters["geometry"]["Ly"]
    parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]
    # lc = ell_ / 5.0

    # Get geometry model
    parameters["geometry"]["geom_type"]
    _N = parameters["model"]["N"]


    # Create the mesh of the specimen with given dimensions
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = os.path.join(outdir, "test_cone")

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    import hashlib
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    # Functional Setting

    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                degree=1)

    element_alpha = ufl.FiniteElement("DG", mesh.ufl_cell(),
                                    degree=0)

    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

    u = dolfinx.fem.Function(V_u, name="Displacement")
    u_ = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")


    alpha = dolfinx.fem.Function(V_alpha, name="Damage")

    # Pack state
    state = {"u": u, "alpha": alpha}

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    # Useful references
    Lx = parameters.get("geometry").get("Lx")

    # Define the state
    u = Function(V_u, name="Unknown")
    u_ = Function(V_u, name="Boundary Unknown")
    zero_u = Function(V_u, name="Boundary Unknown")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    # Boundary sets


    dofs_alpha_left = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], 0.))
    dofs_alpha_right = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    # Boundary data

    u_.interpolate(lambda x: np.ones_like(x[0]))

    # Bounds (nontrivial)

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    # Set Bcs Function
    zero_u.interpolate(lambda x: np.zeros_like(x[0]))
    u_.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, u_, alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

    bc_u_left = dirichletbc(
        np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(
        u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    # Define the model

    bounds={"lb": alpha_lb, "ub": alpha_ub}

    # Material behaviour
    return

def main(custom_parameters):

    print("")
    
    ColorPrint.print_info(
            f"This is a cone-constrained solver!"
        )
    print("")
    print("")
    with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # solvers, cts = ConstrainedEvolution(parameters)
    #  
    from test_discreteDamage import mesh, solver, stability, history_data, loads, stress
    from test_discreteDamage import state, u_, bounds
    from test_discreteDamage import damage_energy_density, elastic_energy_density, dx


    _nameExp = parameters["geometry"]["geom_type"]

    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = os.path.join(outdir, "test_discrete-damage")

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)


    import collections.abc
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    update(parameters, custom_parameters)

    import hashlib
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    u = state["u"]
    alpha = state["alpha"]

    alpha_lb, alpha_ub = bounds["lb"], bounds["ub"]
    _stability = []

    # Loading loop
    for i_t, t in enumerate(loads):
        state_update(alpha, alpha_lb, u_, t)

        logging.critical(f"-- Solving for t = {t:3.2f} --")

        solver.solve()

        parameters["stability"]["cone"]["maxmodes"]

        is_stable = stability.solve(alpha_lb)
        is_elastic = stability.is_elastic()
        inertia = stability.get_inertia()
        # stability.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")
        _stability.append(is_stable)

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State is stable: {is_stable}")

        # cone._solve(alpha_lb, maxmodes)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        _F = assemble_scalar(form(stress(state)))

        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["total_energy"].append(elastic_energy+fracture_energy)
        history_data["solver_data"].append(solver.data)
        history_data["eigs"].append(stability.data["eigs"])
        history_data["stable"].append(stability.data["stable"])
        history_data["F"].append(_F)

        with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    print(history_data)

    from utils.plots import plot_energies, plot_AMit_load, plot_force_displacement

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
        plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")

def state_update(alpha, alpha_lb, u_, t):
    u_.interpolate(lambda x: t * np.ones_like(x[0]))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)

        # update the lower bound
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )


    # Viz

if __name__ == "__main__":

    custom_parameters = {
        "geometry": {
            "geom_type": "discrete-damage"
        },
        "loading": {
            "steps": 10
            },
        "stability": {
            "cone": {            
                "maxmodes": 3,
                "cone_atol": 1.e-7,
                "cone_rtol": 1.e-7,
                "cone_max_it": 100,
                }
        }
    }

    setup(custom_parameters)

    main(custom_parameters)
