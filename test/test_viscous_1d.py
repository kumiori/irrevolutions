#!/usr/bin/env python3
import hashlib
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import os
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
import numpy as np
sys.path.append("../")
import matplotlib.pyplot as plt

from models import DamageElasticityModel
from algorithms.am import AlternateMinimisation, HybridFractureSolver
from algorithms.so import BifurcationSolver, StabilitySolver, BifurcationSolver
from algorithms.ls import LineSearch
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2, seminorm_H1

from meshes.primitives import mesh_bar_gmshapi
from dolfinx.common import Timer, list_timings, TimingType

from solvers.function import vec_to_functions
from utils import simulation_info

import logging

logging.basicConfig(level=logging.INFO)

import dolfinx
import dolfinx.plot
from dolfinx.io import XDMFFile, gmshio
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
import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

comm = MPI.COMM_WORLD

size = comm.Get_size()

from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import form
from solvers.function import vec_to_functions

class BrittleJump1D(DamageElasticityModel):
    """This model accounts for the jump energy across...jumps"""

    def jump_energy_density(self, state, alphadot):

        mesh = state['u'].function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        energy = self.total_energy_density(state) * dx
        L2 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
        
        f_plus = Function(L2)
        f = Function(L2)
        
        alphadot_norm = norm_L2(alphadot)
        
        _F = ufl.derivative(
                energy,
                state["alpha"],
                ufl.TestFunction(state["alpha"].ufl_function_space())
        )
        
        assemble_vector(f.vector, form(_F))
        F_plus, F_minus = self._get_signed_components(f.vector)
        f_plus.interpolate(F_plus)
        
        # psi(f) = sup <-f, β>, β ∈ {β ∈ H^1(Ω), β ≤ 0 : ||β||_{L^2(Ω)} ≤ 1}
        # a good candidate is β = f^+/||f||_{L^2(Ω)}
        
        psi = ufl.dot(f_plus, f) / norm_L2(f)

        return alphadot_norm * psi

    def _get_signed_components(self, f: PETSc.Vec):
        """
        Returns signed components of a vector field
        """
        f_plus = f.copy()
        f_minus = f.copy()
        
        with f.localForm() as f_local \
                , f_plus.localForm() as f_plus_local \
                , f_minus.localForm() as f_minus_local:
            f_plus_local.array = np.maximum(f_local.array, 0)
            f_minus_local.array = np.minimum(f_local.array, 0)
            
        for field in [f_plus, f_minus]:
            field.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
        
        return f_plus, f_minus

    def elastic_energy_density_strain(self, eps, alpha):
        mu = self.mu

        energy_density = (
            self.a(alpha) * 1.0 / 2.0 *
            (2 * mu * ufl.inner(eps, eps) ))
        return energy_density


# Mesh on node model_rank and then distribute
model_rank = 0

def test_viscous_firstorder(parameters, storage):
    # Adapting
    # Calc. Var. (2016) 55:17
    # DOI 10.1007/s00526-015-0947-6

    petsc4py.init(sys.argv)
    comm = MPI.COMM_WORLD

    model_rank = 0

    Lx = parameters["geometry"]["Lx"]
    # Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]
    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]  
    Ly = lc/2
    geom_type = parameters["geometry"]["geom_type"]
    parameters["model"]["model_type"] = '1D'
    
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 30)

    # gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)
    # mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir = "output"
    if storage is None:
        prefix = os.path.join(outdir, "traction_AT2_cone", signature)
    else:
        prefix = storage
    
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    parameters = {**simulation_info, **parameters}

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]
    _nameExp = parameters["geometry"]["geom_type"]
    _nameExp = "bar"
    ell_ = parameters["model"]["ell"]
    lc = parameters["geometry"]["lc"]

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]


    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir = "output"
    if storage is None:
        prefix = os.path.join(outdir, "traction_AT2_cone", signature)
    else:
        prefix = storage
    
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)
        
    # Function spaces
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    # Define the state
    u = Function(V_u, name="Displacement")
    u_ = Function(V_u, name="Boundary Displacement")
    v = Function(V_u, name="Displacement perturbation")
    zero_u = Function(V_u, name="   Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    β = Function(V_alpha, name="Damage perturbation")
    zero_alpha = Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="DamageRate")

    state = {"u": u, "alpha": alpha}
    z = [u, alpha]
    _z = [v, β]
    # need upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="LowerBound")
    alpha_ub = Function(V_alpha, name="UpperBound")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)
    _Omega = assemble_scalar(dolfinx.fem.form(1*dx))

    dofs_alpha_left = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], 0.0))
    dofs_alpha_right = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
    # Set Bcs Function
    zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
    zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
    u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

    bc_u_left = dirichletbc(
        np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(
        u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = [
        dirichletbc(
            np.array(0, dtype=PETSc.ScalarType),
            np.concatenate([dofs_alpha_left, dofs_alpha_right]),
            V_alpha,
        )
    ]
    # bcs_alpha = []

    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    # Define the model

    model = BrittleJump1D(parameters["model"])
    load_par = parameters["loading"]

    # Energy functional
    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    dt = (load_par["max"] - load_par["min"]) / load_par["steps"]
    viscous_coef = parameters["model"]["viscous_eps"] / (2*dt)

    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work \
            + viscous_coef * ufl.dot(alpha - alpha_lb, alpha - alpha_lb) * dx
    
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])

    solver = AlternateMinimisation(
        total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
    )

    hybrid = HybridFractureSolver(
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
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    linesearch = LineSearch(total_energy, state, linesearch_parameters=parameters.get("stability").get("linesearch"))


    history_data = {
        "load": [],
        "elastic_energy": [],
        "jump_energy": [],
        "eps_jump_energy": [],
        "eps_jump_energy_scaled": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
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
        "s": [],
        "s-t": []
    }

    s = load_par["min"]
    jump_energy = 0
    eps_jump_energy = 0
    eps_jump_energy_scaled = 0
    np.set_printoptions(precision=3,suppress=True)

    for i_t, t in enumerate(loads):
        u_.interpolate(lambda x: (t * np.ones_like(x[0]),  np.zeros_like(x[1])))
        u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        ColorPrint.print_bold(f"   Solving first order: AM   ")
        ColorPrint.print_bold(f"===================-=========")

        logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

        solver.solve()

        ColorPrint.print_bold(f"   Solving first order: Hybrid   ")
        ColorPrint.print_bold(f"===================-=============")

        logging.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
        hybrid.solve(alpha_lb)

        # compute the rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
        
        rate_12_norm = 1/_Omega * hybrid.scaled_rate_norm(alphadot, parameters)
        urate_12_norm = 1/_Omega * hybrid.unscaled_rate_norm(alphadot)

        # Compute time
        # s + \int_0^t ||\dot \alpha||_H^1 ds
        # s += ||\dot \alpha||_H^1 dt
        rates = np.array(history_data["rate_12_norm"])
        times = np.array(history_data["load"])
        s_i = rate_12_norm * dt
        s = t + np.trapz(rates, times) + s_i

        ColorPrint.print_bold(f"   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold(f"===================-=================")

        # n_eigenvalues = 10
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = bifurcation.is_elastic()
        # is_critical = bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()
        
        ColorPrint.print_bold(f"   Solving second order: Cone Pb.    ")
        ColorPrint.print_bold(f"===================-=================")
        
        stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum, inertia = inertia)
        max_continuation_iterations = 10
        _continuation_iterations = 0

        # while not stable and _continuation_iterations < max_continuation_iterations:
        _continuation_iterations = 0
        _perturbation = cone.get_perturbation()

        if _perturbation is not None:
            vec_to_functions(_perturbation, [v, β])

            perturbation = {"v": v, "beta": β}
            interval = linesearch.get_unilateral_interval(state, perturbation)

            order = 4
            h_opt, energies_1d, p, _ = linesearch.search(state, perturbation, interval, m=order)
            # h_rnd, energies_1d, p, _ = linesearch.search(state, perturbation, interval, m=order, method = 'random')
            
            # perturb the state
            # linesearch.perturb(state, perturbation, h_opt)

        # hybrid.solve(alpha_lb)
        # is_path = bifurcation.solve(alpha_lb)

        # compute the rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        rate_12_norm = 1/_Omega * hybrid.scaled_rate_norm(alphadot, parameters)
        urate_12_norm = 1/_Omega * hybrid.unscaled_rate_norm(alphadot)

        # Compute jump energy
        eps_jump_energy_t = parameters["model"]["viscous_eps"] * hybrid.unscaled_rate_norm(alphadot)**2
        eps_jump_energy_scaled_t = parameters["model"]["viscous_eps"] * hybrid.scaled_rate_norm(alphadot, parameters)**2
        
        # Compute time
        # s + \int_0^t ||\dot \alpha||_H^1 ds
        # s += ||\dot \alpha||_H^1 dt
        rates = np.array(history_data["rate_12_norm"])
        times = np.array(history_data["load"])
        s_i = rate_12_norm * dt
        s = t + np.trapz(rates, times) + s_i

        inertia = bifurcation.get_inertia()
        stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum, inertia = inertia)

        _continuation_iterations += 1
        # else:
        #     ColorPrint.print_bold(f"We found, or lost something? State is stable: {stable}")

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        logging.critical(f"alpha vector norm: {alpha.vector.norm()}")
        logging.critical(f"alpha lb norm: {alpha_lb.vector.norm()}")
        logging.critical(f"alphadot norm: {alphadot.vector.norm()}")
        logging.critical(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")
        logging.critical(f"scaled rate state_12 norm: {rate_12_norm}")
        logging.critical(f"unscaled scaled rate state_12 norm: {urate_12_norm}")

        jump_energy_t = comm.allreduce(
            assemble_scalar(form(model.jump_energy_density(state, alphadot) * dx)),
            op=MPI.SUM,
        )

        jump_energy += jump_energy_t * dt    
        eps_jump_energy +=  eps_jump_energy_t * dt    
        eps_jump_energy_scaled += eps_jump_energy_scaled_t * dt
                
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
            assemble_scalar(form(_stress[0] * dx)),
            op=MPI.SUM,
        )

        Fform = form(hybrid.F[1])
        Fv = assemble_vector(Fform)

        _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["jump_energy"].append(jump_energy)
        history_data["eps_jump_energy"].append(eps_jump_energy)
        history_data["eps_jump_energy_scaled"].append(eps_jump_energy_scaled)
        history_data["total_energy"].append(elastic_energy+fracture_energy)
        history_data["solver_data"].append(solver.data)
        history_data["eigs"].append(bifurcation.data["eigs"])
        history_data["F"].append(stress)
        history_data["cone_data"].append(cone.data)
        history_data["alphadot_norm"].append(alphadot.vector.norm())
        history_data["rate_12_norm"].append(rate_12_norm)
        history_data["unscaled_rate_12_norm"].append(urate_12_norm)
        history_data["cone-stable"].append(stable)
        history_data["cone-eig"].append(cone.data["lambda_0"])
        history_data["uniqueness"].append(_unique)
        history_data["inertia"].append(inertia)
        history_data["s"].append(s)
        history_data["s-t"].append(s-t)

        with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        ColorPrint.print_bold(f"   Written timely data.    ")
        print()
        print()
        print()
        print()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    import pandas as pd
    df = pd.DataFrame(history_data)
    print(df.drop(['solver_data', 'cone_data'], axis=1))


    from utils.plots import plot_energies, plot_AMit_load, plot_force_displacement

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
        plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")
        my_plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies-rescaled.pdf", times = history_data["s"])
        
        import matplotlib
        fig, ax1 = matplotlib.pyplot.subplots()
        fig.tight_layout()
        ax1.set_xlabel(r"time", fontsize=12)
        ax1.set_ylabel(r"Time", fontsize=12)

        ax1.plot(history_data["load"], history_data["s"], color="tab:blue", linestyle="-", linewidth=1.0, markersize=4.0, marker="o", label=r"Time")
        ax1.plot(history_data["load"], history_data["load"], color="tab:blue", linestyle="-", linewidth=1.0, markersize=4.0, marker="o", label=r"time")
        fig.savefig(f"{prefix}/{_nameExp}_times-rescaled.pdf")


def my_plot_energies(history_data, title="Evolution", file=None, times=None):
    import matplotlib

    fig, ax1 = matplotlib.pyplot.subplots()

    if title is not None:
        ax1.set_title(title, fontsize=12)

    ax1.set_xlabel(r"Load", fontsize=12)
    ax1.set_ylabel(r"Energies", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    if times is None:
        t = np.array(history_data["load"])
    else:
        t = times
    e_e = np.array(history_data["elastic_energy"])
    e_d = np.array(history_data["fracture_energy"])

    # stress-strain curve
    ax1.plot(
        t,
        e_e,
        color="tab:blue",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="o",
        label=r"Elastic",
    )
    ax1.plot(
        t,
        e_d,
        color="tab:red",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        marker="^",
        label=r"Fracture",
    )
    ax1.plot(t, e_d + e_e, color="black", linestyle="-", linewidth=1.0, label=r"Total")

    ax1.legend(loc="upper left")
    if file is not None:
        fig.savefig(file)
    return fig, ax1


def load_parameters(file_path):
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

    # parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = .0001

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = '2D'
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = .1
    parameters["model"]["k_res"] = 0.
    parameters["loading"]["min"] = .99
    parameters["loading"]["max"] = 1.01
    parameters["loading"]["steps"] = 2

    parameters["geometry"]["mesh_size_factor"] = 4

    parameters["model"]["viscous_eps"] = 5.e-3

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature

if __name__ == "__main__":
    # test_NLB(nest=False)

    parameters, signature = load_parameters("../test/parameters.yml")
    ColorPrint.print_bold(f"===================-{signature}-=================")
    _storage = f"output/test_viscous_relaxation/{signature}"
    ColorPrint.print_bold(f"===================-{_storage}-=================")
    # __import__('pdb').set_trace()
    test_viscous_firstorder(parameters, _storage)
    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")
