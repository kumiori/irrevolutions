#!/usr/bin/env python3
from irrevolutions.utils import ColorPrint
from utils.viz import plot_scalar, plot_vector
from utils.plots import plot_energies, plot_force_displacement
from pyvista.plotting.utilities import xvfb
from models import DamageElasticityModel as Brittle
from meshes.primitives import mesh_bar_gmshapi
from algorithms.so import BifurcationSolver, StabilitySolver
from algorithms.am import AlternateMinimisation, HybridSolver
import pyvista
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import pandas as pd
import petsc4py
import ufl
import yaml
from dolfinx.common import list_timings
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
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl

sys.path.append("../")


class BrittleAT2(Brittle):
    """Brittle AT_2 model, without an elastic phase. For fun only."""

    def w(self, alpha):
        """
        Return the dissipated energy function as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function
        return self.w1 * alpha**2


class ResultsStorage:
    """
    Class for storing and saving simulation results.
    """

    def __init__(self, comm, prefix):
        self.comm = comm
        self.prefix = prefix

    def store_results(self, parameters, history_data, state):
        """
        Store simulation results in XDMF and JSON formats.

        Args:
            history_data (dict): Dictionary containing simulation data.
        """
        history_data["load"][-1]

        state["u"]
        state["alpha"]

        if self.comm.rank == 0:
            with open(f"{self.prefix}/parameters.yaml", "w") as file:
                yaml.dump(parameters, file)

        # with XDMFFile(self.comm, f"{self.prefix}/simulation_results.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        #     # for t, data in history_data.items():
        #         # file.write_scalar(data, t)
        #     file.write_mesh(u.function_space.mesh)

        #     file.write_function(u, t)
        #     file.write_function(alpha, t)

        if self.comm.rank == 0:
            with open(f"{self.prefix}/time_data.json", "w") as file:
                json.dump(history_data, file)


# Visualization functions/classes


class Visualization:
    """
    Class for visualizing simulation results.
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def visualise_results(self, history_data):
        """
        Visualise simulation results using appropriate visualization libraries.

        Args:
            history_data (dict): Dictionary containing simulation data.
        """
        # Implement visualization code here

    def save_table(self, data, name):
        """
        Save pandas table results using json.

        Args:
            data (dict): Pandas table containing simulation data.
            name (str): Filename.
        """

        if MPI.COMM_WORLD.rank == 0:
            a_file = open(f"{self.prefix}/{name}.json", "w")
            json.dump(data.to_json(), a_file)
            a_file.close()

    def save_json(self, data, name):
        """
        Save json results.

        Args:
            data (dict): json containing simulation data.
            name (str): filename.
        """

        if MPI.COMM_WORLD.rank == 0:
            a_file = open(f"{self.prefix}/{name}.json", "w")
            json.dump(data, a_file)
            a_file.close()


def main(parameters, model="at2", storage=None):
    petsc4py.init(sys.argv)
    comm = MPI.COMM_WORLD

    model_rank = 0

    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]
    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]
    _lc = ell_ / parameters["geometry"]["ell_lc"]
    geom_type = parameters["geometry"]["geom_type"]

    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, _lc, tdim)
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    outdir = os.path.join(os.path.dirname(__file__), "output")

    if storage is None:
        prefix = os.path.join(outdir, "traction_AT2_cone", signature)
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

    # with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    #     file.write_mesh(mesh)

    element_u = basix.ufl.element(
        "Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,)
    )
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    u = Function(V_u, name="Displacement")
    u_ = Function(V_u, name="Boundary Displacement")
    zero_u = Function(V_u, name="Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    zero_alpha = Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

    state = {"u": u, "alpha": alpha}
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
    zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
    u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bc_u_left = dirichletbc(np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)
    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]
    bcs_alpha = []

    set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
    alpha_ub.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    if model == "at2":
        model = BrittleAT2(parameters["model"])
    elif model == "at1":
        model = Brittle(parameters["model"])
    else:
        raise ValueError("Model not implemented")

    state = {"u": u, "alpha": alpha}

    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    AlternateMinimisation(
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
        # "solver_data": [],
        "cone_data": [],
        "cone-eig": [],
        "eigs": [],
        "uniqueness": [],
        "inertia": [],
        "F": [],
        "alphadot_norm": [],
        "rate_12_norm": [],
        "unscaled_rate_12_norm": [],
        "cone-stable": [],
    }

    check_stability = []

    for i_t, t in enumerate(loads):
        logging.getLogger().setLevel(logging.WARNING)

        u_.interpolate(lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1])))
        u_.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        ColorPrint.print_bold("   Solving first order: AM   ")
        ColorPrint.print_bold("===================-=========")

        logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
        # solver.solve()

        ColorPrint.print_bold("   Solving first order: Hybrid   ")
        ColorPrint.print_bold("===================-=============")

        logging.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
        hybrid.solve(alpha_lb)

        alpha.x.petsc_vec.copy(alphadot.x.petsc_vec)
        alphadot.x.petsc_vec.axpy(-1, alpha_lb.x.petsc_vec)
        alphadot.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.critical(f"alpha vector norm: {alpha.x.petsc_vec.norm()}")
        logging.critical(f"alpha lb norm: {alpha_lb.x.petsc_vec.norm()}")
        logging.critical(f"alphadot norm: {alphadot.x.petsc_vec.norm()}")
        # logging.critical(
        #     f"vector norms [u, alpha]: {[zi.x.petsc_vec.norm() for zi in z]}")

        rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
        urate_12_norm = hybrid.unscaled_rate_norm(alpha)
        logging.critical(f"scaled rate state_12 norm: {rate_12_norm}")
        logging.critical(f"unscaled scaled rate state_12 norm: {urate_12_norm}")

        ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold("===================-=================")

        is_stable = bifurcation.solve(alpha_lb)
        is_elastic = bifurcation.is_elastic()
        inertia = bifurcation.get_inertia()

        check_stability.append(is_stable)

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")

        ColorPrint.print_bold("   Solving second order: Cone Pb.    ")
        ColorPrint.print_bold("===================-=================")

        stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        _stress = model.stress(model.eps(u), alpha)

        stress = comm.allreduce(
            assemble_scalar(form(_stress[0, 0] * dx)),
            op=MPI.SUM,
        )
        _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        # history_data["solver_data"].append(solver.data)
        history_data["eigs"].append(bifurcation.data["eigs"])
        history_data["F"].append(stress)
        history_data["cone_data"].append(cone.data)
        history_data["alphadot_norm"].append(alphadot.x.petsc_vec.norm())
        history_data["rate_12_norm"].append(rate_12_norm)
        history_data["unscaled_rate_12_norm"].append(urate_12_norm)
        history_data["cone-stable"].append(stable)
        history_data["cone-eig"].append(cone.data["lambda_0"])
        history_data["uniqueness"].append(_unique)
        history_data["inertia"].append(inertia)

        # with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        #     file.write_function(u, t)
        #     file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        ColorPrint.print_bold("   Written timely data.    ")

    df = pd.DataFrame(history_data)
    # print(df.drop(['solver_data', 'cone_data'], axis=1))
    print(df.drop(["cone_data"], axis=1))

    with dolfinx.common.Timer("~Postprocessing and Vis"):
        if comm.rank == 0:
            plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
            # plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
            plot_force_displacement(
                history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
            )

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        plotter = pyvista.Plotter(
            title="Traction test",
            window_size=[1600, 600],
            shape=(1, 2),
        )
        _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
        _plt = plot_vector(u, plotter, subplot=(0, 1))
        _plt.screenshot(f"{prefix}/traction-state.png")

    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold("   Done!    ")

    performance = {
        "N": [],
        "dofs": [],
    }
    performance["N"].append(MPI.COMM_WORLD.size)
    performance["dofs"].append(
        sum([V.dofmap.bs * V.dofmap.index_map.size_global for V in [V_u, V_alpha]])
    )

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    return history_data, performance, state


# Configuration handling (load parameters from YAML)


def load_parameters(file_path, model="at2"):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    if model == "at2":
        parameters["loading"]["min"] = 0.9
        parameters["loading"]["max"] = 0.9
        parameters["loading"]["steps"] = 1

    elif model == "at1":
        parameters["loading"]["min"] = 1.03
        parameters["loading"]["max"] = 1.03
        parameters["loading"]["steps"] = 1

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def param_vs_ell():
    ell_list = [0.1, 0.2]
    # __import__('pdb').set_trace()

    for ell in ell_list:
        parameters, signature = parameters_vs_ell(parameters=base_parameters, ell=ell)
        _storage = f"output/parametric/traction-bar/vs_ell/{base_signature}/{signature}"
        history_data, performance, state = main(parameters, _storage)


def param_vs_s(base_parameters, base_signature):
    # s_list = [1.e-08, 1.e-07, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
    s_list = np.logspace(-9, -1, 9).tolist()

    from irrevolutions.utils import table_timing_data

    _rootdir = f"output/parametric/traction-bar/vs_s/{base_signature}"

    if MPI.COMM_WORLD.rank == 0:
        Path(_rootdir).mkdir(parents=True, exist_ok=True)

    if MPI.COMM_WORLD.rank == 0:
        with open(f"{_rootdir}/parameters.yaml", "w") as file:
            yaml.dump(base_parameters, file)

    for s in s_list:
        parameters, signature = parameters_vs_SPA_scaling(
            parameters=base_parameters, s=s
        )
        _storage = f"output/parametric/traction-bar/vs_s/{base_signature}/{signature}"

        ColorPrint.print_bold(
            f"===PARAMETRIC vs S================-{s}-================="
        )
        ColorPrint.print_bold(f"===================-{signature}-=================")

        with dolfinx.common.Timer("~Computation Experiment"):
            history_data, performance, state = main(parameters, _storage)

        _timings = table_timing_data()

        storage = ResultsStorage(MPI.COMM_WORLD, _storage)

        storage.store_results(parameters, history_data, state)

        visualization = Visualization(_storage)

        visualization.visualise_results(history_data)

        visualization.save_table(pd.DataFrame(history_data), "history")
        visualization.save_table(_timings, "timing_data")

        visualization.save_json(performance, "performance")

    ColorPrint.print_bold(f"===================-{signature}-=================")

    # Store and visualise results

    return history_data, _timings


def param_vs_dry(base_parameters, base_signature):
    # s_list = [1.e-08, 1.e-07, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
    s_list = np.arange(0, 10).tolist()

    from irrevolutions.utils import table_timing_data

    _rootdir = f"output/parametric/traction-bar/vs_s/{base_signature}"

    if MPI.COMM_WORLD.rank == 0:
        Path(_rootdir).mkdir(parents=True, exist_ok=True)

    if MPI.COMM_WORLD.rank == 0:
        with open(f"{_rootdir}/parameters.yaml", "w") as file:
            yaml.dump(base_parameters, file)

    for s in s_list:
        # parameters, signature = parameters_vs_SPA_scaling(parameters=base_parameters, s=s)
        parameters = base_parameters
        signature = s
        _storage = f"output/parametric/traction-bar/dry/{signature}"

        ColorPrint.print_bold(
            f"===PARAMETRIC vs S================-{s}-================="
        )
        ColorPrint.print_bold(f"===================-{signature}-=================")

        with dolfinx.common.Timer("~Computation Experiment"):
            history_data, performance, state = main(parameters, _storage)

        _timings = table_timing_data()

        storage = ResultsStorage(MPI.COMM_WORLD, _storage)

        storage.store_results(parameters, history_data, state)

        visualization = Visualization(_storage)

        visualization.visualise_results(history_data)

        visualization.save_table(pd.DataFrame(history_data), "history")
        visualization.save_table(_timings, "timing_data")

        visualization.save_json(performance, "performance")

    ColorPrint.print_bold(f"===================-{signature}-=================")

    # Store and visualise results

    return history_data, _timings


if __name__ == "__main__":
    import argparse

    from utils.parametric import (
        parameters_vs_ell,
        parameters_vs_n_refinement,
        parameters_vs_SPA_scaling,
    )

    admissible_models = {"at1", "at2", "thinfilm"}

    parser = argparse.ArgumentParser(description="Process evolution.")

    parser.add_argument("-s", type=str, default=1e-4, help="scaling")

    parser.add_argument("-n", type=int, default=3, help="resolution: ell to h ratio")

    parser.add_argument("--model", choices=admissible_models, help="The model to use.")

    args = parser.parse_args()

    base_parameters, base_signature = load_parameters(
        "../test/parameters.yml", model=args.model
    )

    if "-s" in sys.argv:
        parameters, signature = parameters_vs_SPA_scaling(
            parameters=base_parameters, s=np.float(args.s)
        )
        _storage = f"output/parametric/traction-bar/vs_s/{args.model}/{base_signature}/{signature}"

    elif "-n" in sys.argv:
        parameters, signature = parameters_vs_n_refinement(
            parameters=base_parameters, r=np.int(args.n)
        )
        _storage = f"output/parametric/traction-bar/vs_resolution/{args.model}/{base_signature}/{signature}"

    else:
        parameters, signature = base_parameters, base_signature
        _storage = f"output/parametric/traction-bar/vs_s/{args.model}/base/{signature}"

    ColorPrint.print_bold("   Base    ")
    ColorPrint.print_bold(f"===================-model {args.model}-=================")
    ColorPrint.print_bold(f"===================-{base_signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    print(json.dumps(parameters, indent=2))

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        history_data, performance, state = main(parameters, args.model, _storage)

    # Store and visualise results
    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    storage.store_results(parameters, history_data, state)

    visualization = Visualization(_storage)

    visualization.visualise_results(history_data)

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================-{signature}-=================")

    # timings

    from irrevolutions.utils import table_timing_data

    _timings = table_timing_data()

    visualization.save_table(pd.DataFrame(history_data), "_history_data.json")
    visualization.save_table(_timings, "timing_data")
    visualization.save_json(performance, "performance")
