#!/usr/bin/env python3
from test_discreteDamage import ConeSolver
from solvers.function import functions_to_vec
from utils.plots import plot_energies
from utils import ColorPrint, norm_H1, norm_L2
from solvers import SNESSolver
from meshes.primitives import mesh_bar_gmshapi
from algorithms.so import StabilitySolver
import json
import logging
import os
import pdb
import sys
from pathlib import Path

from pip import main

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pandas as pd
import petsc4py
import ufl
import yaml
from dolfinx import log
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form, locate_dofs_geometrical, set_bc)
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import CellType
from mpi4py import MPI
comm = MPI.COMM_WORLD

from petsc4py import PETSc
from sympy import derive_by_array

sys.path.append("../")

sys.path.append("../")


"""Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""


logging.getLogger().setLevel(logging.CRITICAL)


class ConstrainedProblem:
    """A Problem, solved.
        We consider the following problem:
        ...
    """

    # has: context and solver(s)


def main(custom_parameters):
    
    from test_discreteDamage import mesh, solver, stability, history_data, loads, stress
    from test_discreteDamage import state, u_, bounds
    from test_discreteDamage import damage_energy_density, elastic_energy_density, dx

    print("")
    
    ColorPrint.print_info(
            f"This is a cone-constrained solver!"
        )
    print("")
    print("")
    with open("parameters.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    _nameExp = parameters["geometry"]["geom_type"]

    outdir = "output"
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

        maxmodes = parameters["stability"]["cone"]["maxmodes"]

        is_stable = stability.solve(alpha_lb, maxmodes)
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
    # print(flatten(_parameters))
    main(custom_parameters)
