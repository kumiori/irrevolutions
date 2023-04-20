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

from dolfinx.fem.petsc import (
    set_bc,
    assemble_vector
    )
from dolfinx.io import XDMFFile, gmshio
import logging
from dolfinx.common import Timer, list_timings, TimingType

sys.path.append("../")
from algorithms.so import StabilitySolver, ConeSolver
from solvers import SNESSolver
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2




sys.path.append("../")


"""Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""


from solvers.function import functions_to_vec
logging.getLogger().setLevel(logging.CRITICAL)

comm = MPI.COMM_WORLD

class _AlternateMinimisation:
    def __init__(self,
                total_energy,
                state,
                bcs,
                solver_parameters={},
                bounds=(dolfinx.fem.function.Function,
                        dolfinx.fem.function.Function)
                ):
        self.state = state
        self.alpha = state["alpha"]
        self.alpha_old = dolfinx.fem.function.Function(self.alpha.function_space)
        self.u = state["u"]
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]
        self.total_energy = total_energy
        self.solver_parameters = solver_parameters

        V_u = state["u"].function_space
        V_alpha = state["alpha"].function_space

        energy_u = ufl.derivative(
            self.total_energy, self.u, ufl.TestFunction(V_u))
        energy_alpha = ufl.derivative(
            self.total_energy, self.alpha, ufl.TestFunction(V_alpha)
        )

        self.F = [energy_u, energy_alpha]

        self.elasticity = SNESSolver(
            energy_u,
            self.u,
            bcs.get("bcs_u"),
            bounds=None,
            petsc_options=self.solver_parameters.get("elasticity").get("snes"),
            prefix=self.solver_parameters.get("elasticity").get("prefix"),
        )

        self.damage = SNESSolver(
            energy_alpha,
            self.alpha,
            bcs.get("bcs_alpha"),
            bounds=(self.alpha_lb, self.alpha_ub),
            petsc_options=self.solver_parameters.get("damage").get("snes"),
            prefix=self.solver_parameters.get("damage").get("prefix"),
        )

    def solve(self, outdir=None):

        alpha_diff = dolfinx.fem.Function(self.alpha.function_space)

        self.data = {
            "iteration": [],
            "error_alpha_L2": [],
            "error_alpha_H1": [],
            "F_norm": [],
            "error_alpha_max": [],
            "error_residual_F": [],
            "solver_alpha_reason": [],
            "solver_alpha_it": [],
            "solver_u_reason": [],
            "solver_u_it": [],
            "total_energy": [],
        }
        for iteration in range(
            self.solver_parameters.get("damage_elasticity").get("max_it")
        ):
            with dolfinx.common.Timer("~Alternate Minimization : Elastic solver"):
                (solver_u_it, solver_u_reason) = self.elasticity.solve()
            with dolfinx.common.Timer("~Alternate Minimization : Damage solver"):
                (solver_alpha_it, solver_alpha_reason) = self.damage.solve()

            # Define error function
            self.alpha.vector.copy(alpha_diff.vector)
            alpha_diff.vector.axpy(-1, self.alpha_old.vector)
            alpha_diff.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            error_alpha_H1 = norm_H1(alpha_diff)
            error_alpha_L2 = norm_L2(alpha_diff)

            Fv = [assemble_vector(form(F)) for F in self.F]

            Fnorm = np.sqrt(
                np.array(
                    [comm.allreduce(Fvi.norm(), op=MPI.SUM)
                        for Fvi in Fv]
                ).sum()
            )

            error_alpha_max = alpha_diff.vector.max()[1]
            total_energy_int = comm.allreduce(
                assemble_scalar(form(self.total_energy)), op=MPI.SUM
            )
            residual_F = assemble_vector(self.elasticity.F_form)
            residual_F.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(residual_F, self.elasticity.bcs, self.u.vector)
            error_residual_F = ufl.sqrt(residual_F.dot(residual_F))

            self.alpha.vector.copy(self.alpha_old.vector)
            self.alpha_old.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, res F Error: {error_residual_F:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, H1 Error: {error_alpha_H1:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, L2 Error: {error_alpha_L2:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, Linfty Error: {error_alpha_max:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            self.data["iteration"].append(iteration)
            self.data["error_alpha_L2"].append(error_alpha_L2)
            self.data["error_alpha_H1"].append(error_alpha_H1)
            self.data["F_norm"].append(Fnorm)
            self.data["error_alpha_max"].append(error_alpha_max)
            self.data["error_residual_F"].append(error_residual_F)
            self.data["solver_alpha_it"].append(solver_alpha_it)
            self.data["solver_alpha_reason"].append(solver_alpha_reason)
            self.data["solver_u_reason"].append(solver_u_reason)
            self.data["solver_u_it"].append(solver_u_it)
            self.data["total_energy"].append(total_energy_int)


            if (
                self.solver_parameters.get(
                    "damage_elasticity").get("criterion")
                == "residual_u"
            ):
                if error_residual_F <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
            if (
                self.solver_parameters.get(
                    "damage_elasticity").get("criterion")
                == "alpha_H1"
            ):
                if error_alpha_H1 <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}"
            )


petsc4py.init(sys.argv)

def discrete_atk(arg_N=2):

    # Mesh on node model_rank and then distribute
    model_rank = 0


    with open("./parameters.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # parameters["stability"]["cone"] = ""
    # parameters["cone"]["atol"] = 1e-7

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = '1D'
    parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1
    parameters["model"]["k_res"] = 1e-4
    parameters["model"]["k"] = 3
    parameters["model"]["N"] = arg_N
    # parameters["loading"]["max"] = 2.
    parameters["loading"]["max"] = parameters["model"]["k"] 
    parameters["loading"]["steps"] = 100

    parameters["geometry"]["geom_type"] = "discrete-damageable"
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]
    # lc = ell_ / 5.0

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]
    _N = parameters["model"]["N"]


    # Create the mesh of the specimen with given dimensions
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

    import hashlib
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    outdir = "output"
    prefix = os.path.join(outdir, f"discrete-atk-N{parameters['model']['N']}")

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    _crunchdir = os.path.join(outdir, "discrete-atk-sigs")
    if comm.rank == 0:
        Path(_crunchdir).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    if comm.rank == 0:
        with open(f"{_crunchdir}/{signature}.md5", 'w') as f:
            f.write('')

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

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

    # Material behaviour

    # mat_par = parameters.get()


    def a(alpha):
        k_res = parameters["model"]['k_res']
        return (1 - alpha)**2 + k_res


    def a_atk(alpha):
        k_res = parameters["model"]['k_res']
        _k = parameters["model"]['k']
        return (1 - alpha) / ((_k-1) * alpha + 1)


    def w(alpha):
        """
        Return the homogeneous damage energy term,
        as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function
        return alpha


    def elastic_energy_density_atk(state):
        """
        Returns the elastic energy density from the state.
        """
        # Parameters
        _mu = parameters["model"]['mu']
        _N = parameters["model"]['N']

        alpha = state["alpha"]
        u = state["u"]
        eps = ufl.grad(u) 

        energy_density = _mu / 2. * a_atk(alpha) * ufl.inner(eps, eps)
        return energy_density


    def damage_energy_density(state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        _mu = parameters["model"]["mu"]
        _w1 = parameters["model"]["w1"]
        _ell = parameters["model"]["ell"]
        # Get the damage
        alpha = state["alpha"]
        # Compute the damage gradient
        grad_alpha = ufl.grad(alpha)
        # Compute the damage dissipation density
        D_d = _w1 * w(alpha) + _w1 * _ell**2 * ufl.dot(
            grad_alpha, grad_alpha)
        return D_d


    def stress(state):
        """
        Return the one-dimensional stress
        """
        u = state["u"]
        alpha = state["alpha"]

        return parameters["model"]['mu'] * a_atk(alpha) * u.dx() * dx

    total_energy = (elastic_energy_density_atk(state) +
                    damage_energy_density(state)) * dx

    # Energy functional
    # f = Constant(mesh, 0)
    f = Constant(mesh, np.array(0, dtype=PETSc.ScalarType))

    external_work = f * state["u"] * dx

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])

    solver = _AlternateMinimisation(
        total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
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
        "eigs": [],
        "cone-stable": [],
        "non-bifurcation": [],
        "F": [],
        "alpha_t": [],
        "u_t": [],
    }

    check_stability = []

    logging.basicConfig(level=logging.INFO)

    for i_t, t in enumerate(loads):
        u_.interpolate(lambda x: t * np.ones_like(x[0]))
        u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.critical(f"-- Solving for t = {t:3.2f} --")

        solver.solve()

        # n_eigenvalues = 10
        is_stable = stability.solve(alpha_lb)
        is_elastic = stability.is_elastic()
        inertia = stability.get_inertia()
        # stability.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")
        check_stability.append(is_stable)

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State is stable: {is_stable}")

        stable = cone._solve(alpha_lb)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(elastic_energy_density_atk(state) * dx)),
            op=MPI.SUM,
        )
        _F = assemble_scalar( form(stress(state)) )
        
        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["total_energy"].append(elastic_energy+fracture_energy)
        history_data["solver_data"].append(solver.data)
        history_data["cone_data"].append(cone.data)
        history_data["eigs"].append(stability.data["eigs"])
        history_data["non-bifurcation"].append(not stability.data["stable"])
        history_data["cone-stable"].append(stable)
        history_data["F"].append(_F)
        history_data["alpha_t"].append(state["alpha"].vector.array.tolist())
        history_data["u_t"].append(state["u"].vector.array.tolist())
        
        logging.critical(f"u_t {u.vector.array}")
        logging.critical(f"u_t norm {state['u'].vector.norm()}")

        with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    # print(history_data)

    df = pd.DataFrame(history_data)
    print(df)


    return history_data, prefix, _nameExp

def postprocess(history_data, prefix, nameExp):
    """docstring for postprocess"""
    

    from utils.plots import plot_energies, plot_AMit_load, plot_force_displacement

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{nameExp}_it_load.pdf")
        plot_force_displacement(history_data, file=f"{prefix}/{nameExp}_stress-load.pdf")


    # Viz


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process evolution.')
    parser.add_argument('-N', type=int, default=2,
                        help='Number of elements')

    args = parser.parse_args()
    # print()

    # __import__('pdb').set_trace()
    history_data, prefix, name = discrete_atk(args.N)

    logging.info(f'Output in {prefix}')

    postprocess(history_data, prefix, name)

    logging.info(f'Output in {prefix}')
else:
   print("File executed when imported")