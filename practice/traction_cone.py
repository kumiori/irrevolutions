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
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation
from algorithms.so import StabilitySolver
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

class ConeSolver(StabilitySolver):
    """Base class for a minimal implementation of the solution of eigenvalue
    problems bound to a cone. Based on numerical recipe SPA and KR existence result
    Thanks Yves and Luc."""

    def __init__(
        self,
        energy: ufl.form.Form,
        state: dict,
        bcs: list,
        nullspace=None,
        cone_parameters=None,
    ):
        super(ConeSolver, self).__init__(
            energy,
            state,
            bcs,
            nullspace,
            stability_parameters=cone_parameters,

    )

    def _solve(self, alpha_old: dolfinx.fem.function.Function, neig=None):
        """Recursively solves (until convergence) the abstract eigenproblem
        K \ni x \perp y := Ax - \lambda B x \in K^*
        based on the SPA recipe, cf. ...
        """
        _s = 0.1
        self.iterations = 0
        errors = []
        self.stable = True
        stable = self.solve(alpha_old)
        self.data = {
            "iteration": [],
            "error_x_L2": [],
            "lambda_k": [],
            "y_norm_L2": [],
        }
        
        self._converged = False

        # The cone is non-trivial, aka non-empty
        # only if the state is irreversibly damage-critical

        if self._critical:

            # loop

            _x = dolfinx.fem.petsc.create_vector_block(self.F)        
            _y = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Ax = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Bx = dolfinx.fem.petsc.create_vector_block(self.F)        
            self._xold = dolfinx.fem.petsc.create_vector_block(self.F)    
            
            # Map current solution into vector _x
            functions_to_vec(self.Kspectrum[0].get("xk"), _x)
            
            self.data["lambda_k"].append(self.Kspectrum[0].get("lambda"))

            with dolfinx.common.Timer(f"~Second Order: Cone Solver - SPA s={_s}"):
        
                while not self.loop(_x):
                    errors.append(self.error)
                    # make it admissible: map into the cone
                    # logging.critical(f"_x is in the cone? {self._isin_cone(_x)}")
                    
                    self._cone_project(_x)

                    # logging.critical(f"_x is in the cone? {self._isin_cone(_x)}")
                    # K_t spectrum:
                    # compute {lambdat, xt, yt}

                    if self.eigen.restriction is not None:
                        _A = self.eigen.rA
                        _B = self.eigen.rB

                        _x = self.eigen.restriction.restrict_vector(_x)
                        _y = self.eigen.restriction.restrict_vector(_y)
                        _Ax = self.eigen.restriction.restrict_vector(_Ax)
                        _Bx = self.eigen.restriction.restrict_vector(_Bx)
                    else:
                        _A = self.eigen.A
                        _B = self.eigen.B

                    _A.mult(_x, _Ax)
                    xAx = _x.dot(_Ax)

                    # compute: lmbda_t
                    if not self.eigen.empty_B():
                        _B.mult(_x, _Bx)
                        xBx = _x.dot(_Bx)
                        _lmbda_t = xAx/xBx
                    else:
                        logging.debug("B = Id")
                        _Bx = _x
                        _lmbda_t = xAx / _x.dot(_x)

                    # compute: y_t = _Ax - _lmbda_t * _Bx
                    _y.waxpy(-_lmbda_t, _Bx, _Ax)

                    # construct perturbation
                    # _v = _x - _s*y_t

                    _x.copy(self._xold)
                    _x.axpy(-_s, _y)
                    
                    # project onto cone
                    self._cone_project(_x)
                    
                    # L2-normalise
                    n2 = _x.normalize()
                    # _x.view()
                    # iterate
                    # x_i+1 = _v 

                    self.data["lambda_k"].append(_lmbda_t)
                    self.data["y_norm_L2"].append(_y.norm())
                    
            logging.critical(f"Convergence of SPA algorithm with s={_s}")
            print(errors)
            logging.critical(f"Eigenfunction is in cone? {self._isin_cone(_x)}")
            
            self.data["iterations"] = self.iterations
            self.data["error_x_L2"] = errors

            if (self._isin_cone(_x)):
                # bifurcating out of existence, not out of a numerical test
            # if (self._converged and _lmbda_t < float(self.parameters.get("cone").get("cone_atol"))):
                stable = bool(False)
        return bool(stable)
    
    def convergenceTest(self, x):
        """Test convergence of current iterate x against 
        prior"""
        # _atol = self.parameters.get("eigen").get("eps_tol")
        # _maxit = self.parameters.get("eigen").get("eps_max_it")

        _atol = self.parameters.get("cone").get("cone_atol")
        _maxit = self.parameters.get("cone").get("cone_max_it")

        if self.iterations == _maxit:
            raise RuntimeError(f'SPA solver did not converge within {_maxit} iterations. Aborting')
            # return False        
        # xdiff = -x + x_old
        diff = x.duplicate()
        diff.zeroEntries()

        diff.waxpy(-1., self._xold, x)

        error_x_L2 = diff.norm()
        self.error = error_x_L2
        logging.debug(f"error_x_L2? {error_x_L2}")

        self.data["iteration"].append(self.iterations)
        self.data["error_x_L2"].append(error_x_L2)
        print(error_x_L2)
        if error_x_L2 < _atol:
            self._converged = True
            return True
        elif self.iterations == 0 or error_x_L2 >= _atol:
            return False


    # v = mode[i].get("xk") for mode in self.spectrum
    def loop(self, x):
        # its = self.iterations
        reason = self.convergenceTest(x)
        
        # update xold
        # x.copy(self._xold)
        # x.vector.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )

        if not reason:
            self.iterations += 1

        return reason

    def _isin_cone(self, x):
        """Is in the zone IFF x is in the cone"""

        # get the subvector associated to damage dofs with inactive constraints 
        _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _sub = x.getSubVector(_is)

        return (_sub.array >= 0).all()
        
    def _cone_project(self, v):
        """Projection vector into the cone

            takes arguments:
            - v: vector in a mixed space

            returns
        """
        
        # get the subvector associated to damage dofs with inactive constraints 
        _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _sub = v.getSubVector(_is)
        zero = _sub.duplicate()
        zero.zeroEntries()

        _sub.pointwiseMax(_sub, zero)
        v.restoreSubVector(_is, _sub)
        return

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

with open("../test/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

parameters["cone"] = ""
# parameters["cone"]["atol"] = 1e-7

parameters["model"]["model_dimension"] = 2
parameters["model"]["model_type"] = '1D'
parameters["model"]["w1"] = 1
parameters["model"]["ell"] = .3
parameters["model"]["k_res"] = 1e-8
parameters["loading"]["max"] = 1.8
parameters["loading"]["steps"] = 10

parameters["geometry"]["geom_type"] = "traction-bar"
# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]

_nameExp = parameters["geometry"]["geom_type"]
ell_ = parameters["model"]["ell"]
# lc = ell_ / 5.0

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

outdir = "output"
prefix = os.path.join(outdir, "traction_cone")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, parameters["geometry"]["lc"], tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


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

# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = FunctionSpace(mesh, element_u)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V_alpha = FunctionSpace(mesh, element_alpha)

# Define the state
u = Function(V_u, name="Displacement")
u_ = Function(V_u, name="Boundary Displacement")
zero_u = Function(V_u, name="   Boundary Displacement")
alpha = Function(V_alpha, name="Damage")
zero_alpha = Function(V_alpha, name="Damage Boundary Field")

state = {"u": u, "alpha": alpha}

# need upper/lower bound for the damage field
alpha_lb = Function(V_alpha, name="Lower bound")
alpha_ub = Function(V_alpha, name="Upper bound")

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

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

# import pdb; pdb.set_trace()

bc_u_right = dirichletbc(
    u_, dofs_u_right)
bcs_u = [bc_u_left, bc_u_right]

# bcs_alpha = [
#     dirichletbc(
#         np.array(0, dtype=PETSc.ScalarType),
#         np.concatenate([dofs_alpha_left, dofs_alpha_right]),
#         V_alpha,
#     )
# ]

bcs_alpha = []

set_bc(alpha_ub.vector, bcs_alpha)
alpha_ub.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)


bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
# Define the model

model = Brittle(parameters["model"])

# Pack state
state = {"u": u, "alpha": alpha}

# Material behaviour

# Energy functional
f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
external_work = ufl.dot(f, state["u"]) * dx
total_energy = model.total_energy_density(state) * dx - external_work

load_par = parameters["loading"]
loads = np.linspace(load_par["min"],
                    load_par["max"], load_par["steps"])

solver = AlternateMinimisation(
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
    "stable": [],
    "F": [],
}

check_stability = []

# logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

for i_t, t in enumerate(loads):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]),  np.zeros_like(x[1])))
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
    
    # __import__('pdb').set_trace()
    stable = cone._solve(alpha_lb)
    # _F = assemble_scalar( form(stress(state)) )
    
    fracture_energy = comm.allreduce(
        assemble_scalar(form(model.damage_dissipation_density(state) * dx)),
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
    # __import__('pdb').set_trace()
    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+fracture_energy)
    history_data["solver_data"].append(solver.data)
    history_data["eigs"].append(stability.data["eigs"])
    history_data["stable"].append(stable)
    # history_data["stable"].append(stability.data["stable"])
    history_data["F"].append(stress)
    history_data["cone_data"].append(cone.data)
    
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


from utils.plots import plot_energies, plot_AMit_load, plot_force_displacement

if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
    plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
    plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")


# Viz

# Viz
from pyvista.utilities import xvfb
import pyvista
import sys
from utils.viz import plot_mesh, plot_vector, plot_scalar
# 
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
# if comm.rank == 0:
#     plot_energies(history_data, file=f"{prefix}_energies.pdf")
#     plot_AMit_load(history_data, file=f"{prefix}_it_load.pdf")
