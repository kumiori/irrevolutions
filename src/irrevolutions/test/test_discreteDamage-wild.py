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
from algorithms.so import BifurcationSolver
from solvers import SNESSolver
from meshes.primitives import mesh_bar_gmshapi
from irrevolutions.utils import ColorPrint
from utils.plots import plot_energies
from irrevolutions.utils import norm_H1, norm_L2




sys.path.append("../")


"""
Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""


from solvers.function import functions_to_vec
logging.getLogger().setLevel(logging.CRITICAL)

class StabilitySolver(BifurcationSolver):
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
        super(StabilitySolver, self).__init__(
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
        stable = self.solve(alpha_old, neig)
        
        # The cone is non-trivial, aka non-empty
        # only if the state is irreversibly damage-critical

        if self._critical:

            # loop
            # __import__('pdb').set_trace()

            _x = dolfinx.fem.petsc.create_vector_block(self.F)        
            _y = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Ax = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Bx = dolfinx.fem.petsc.create_vector_block(self.F)        
            self._xold = dolfinx.fem.petsc.create_vector_block(self.F)    
            
            # Map current solution into vector _x
            functions_to_vec(self.Kspectrum[0].get("xk"), _x)
    
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
                    logging.critical("B = Id")
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

            logging.critical(f"Convergence of SPA algorithm with s={_s}")
            print(errors)
            logging.critical(f"eigenfunction is in cone? {self._isin_cone(_x)}")
        
        return stable
    
    def convergenceTest(self, x):
        """Test convergence of current iterate x against 
        prior"""
        _atol = self.parameters.get("eigen").get("eps_tol")
        _maxit = self.parameters.get("eigen").get("eps_max_it")

        if self.iterations == _maxit:
            raise RuntimeError(f'SPA solver did not converge within {_maxit} iterations. Aborting')
            # return False        
        # xdiff = -x + x_old
        diff = x.duplicate()
        diff.zeroEntries()

        diff.waxpy(-1., self._xold, x)

        error_alpha_L2 = diff.norm()
        self.error = error_alpha_L2
        logging.critical(f"error_alpha_L2? {error_alpha_L2}")

        if error_alpha_L2 < _atol:
            return True
        elif self.iterations == 0 or error_alpha_L2 >= _atol:
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
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


def test(comm):
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



    parameters, mesh, V_u, V_alpha, state, alpha_ub, alpha_lb = test(comm)

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


    stability = BifurcationSolver(
        total_energy, state, bcs, stability_parameters=parameters.get("stability")
    )


    cone = StabilitySolver(
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
        "eigs": [],
        "stable": [],
        "F": [],
    }

    check_stability = []

    logging.basicConfig(level=logging.INFO)

    _solvers = (solver, stability, cone)
    _ctx = (mesh, state, bounds, history_data, loads, stress, u_, damage_energy_density, elastic_energy_density, dx)

    return (_solvers, _ctx)

    # Material behaviour

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

def stress(state):
    """
    Return the one-dimensional stress
    """
    u = state["u"]
    alpha = state["alpha"]

    return parameters["model"]['mu'] * a_atk(alpha) * u.dx() * dx

def elastic_energy_density(state):
    """
    Returns the elastic energy density from the state.
    """
    # Parameters
    alpha = state["alpha"]
    u = state["u"]
    eps = ufl.grad(u)

    _mu = parameters["model"]['mu']
    energy_density = a(alpha) * _mu * ufl.inner(eps, eps)
    return energy_density


def elastic_energy_density_atk(state):
    """
    Returns the elastic energy density from the state.
    """
    # Parameters
    alpha = state["alpha"]
    u = state["u"]
    eps = ufl.grad(u)

    _mu = parameters["model"]['mu']
    energy_density = a_atk(alpha) * _mu * ufl.inner(eps, eps)
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

