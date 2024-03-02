import logging
import pdb
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

petsc4py.init(sys.argv)

from mpi4py import MPI
from utils.viz import plot_mesh, plot_vector, plot_scalar

comm = MPI.COMM_WORLD
# import pdb
import dolfinx.plot
from utils import norm_H1, norm_L2
import pandas as pd

# import pyvista
import yaml
from algorithms.am import AlternateMinimisation as AM, HybridFractureSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from models import DamageElasticityModel as Brittle
from utils import ColorPrint, set_vector_to_constant
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary, CellType, create_rectangle
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import (
    set_bc,
    assemble_vector
    )

import pyvista
from pyvista.utilities import xvfb
from solvers import SNESSolver

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

from utils.viz import plot_matrix

import solvers.restriction as restriction



class NLBSolver(StabilitySolver):
    """Base class for a minimal implementation of the solution of eigenvalue
    problems bound to a cone. Based on numerical recipe SPA and KR existence result
    Thanks Yves and Luc."""

    def __init__(
        self,
        energy: ufl.form.Form,
        state: dict,
        bcs: list,
        nullspace=None,
        nlb_parameters=None,
    ):
        self.state = [state["alpha"]]
        alpha = self.state[0]
        self.parameters = nlb_parameters

        # self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space

        self.mesh = alpha.function_space.mesh

        L = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))
        self.lmbda0 = dolfinx.fem.Function(L)

        # this should be a function of the entire state
        self.energy = energy

        self.F_ = [
            ufl.derivative(
                energy,
                state["alpha"],
                ufl.TestFunction(state["alpha"].ufl_function_space()),
            ),
        ]
        self.F = dolfinx.fem.form(self.F_)
        
        # Is the current state critical? 
        self._critical = False

        self.bcs = bcs["bcs_alpha"]
        pass

    def get_inactive_dofset(self, a_old) -> set:
        """Computes the set of dofs where damage constraints are inactive
        based on the energy gradient and the ub constraint. The global
        set of inactive constraint-dofs is the union of constrained
        alpha-dofs and u-dofs.
        """
        gtol = self.parameters.get("inactiveset_gatol")
        pwtol = self.parameters.get("inactiveset_pwtol")
        V_u = self.state[0].function_space

        F = dolfinx.fem.petsc.assemble_vector(self.F[0])
        
        print('Fa', F[:])
        
        with F.localForm() as f_local:
            idx_grad_local = np.where(np.isclose(f_local[:], 0.0, atol=gtol))[0]

        with self.state[
            0
        ].vector.localForm() as a_local, a_old.vector.localForm() as a_old_local:
            idx_ub_local = np.where(np.isclose(a_local[:], 1.0, rtol=pwtol))[0]
            idx_lb_local = np.where(np.isclose(a_local[:], a_old_local[:], rtol=pwtol))[
                0
            ]

        idx_alpha_local = set(idx_grad_local).difference(
            set().union(idx_ub_local, idx_lb_local)
        )

        # V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
        # dofs_u_all = np.arange(V_u_size, dtype=np.int32)

        dofs_alpha_inactive = np.array(list(idx_alpha_local), dtype=np.int32)

        if len(dofs_alpha_inactive) > 0:
            self._critical = True
        else:
            self._critical = False
        
        logging.critical(
            f"rank {comm.rank}) Current state is damage-critical? {self._critical}"
        )
        if self._critical:
            logging.critical(
                f"rank {comm.rank})     > The cone is open 🍦"
            )

        # F.view()

        restricted_dofs = [dofs_alpha_inactive]

        localSize = F.getLocalSize()

        restricted = len(dofs_alpha_inactive)

        logging.debug(
            f"rank {comm.rank}) Restricted to (local) {restricted}/{localSize} nodes, {float(restricted/localSize):.1%} (local)",
        )

        return restricted_dofs


    def _potential_energy(self, equilibrium: dict):
        """returns total_energy computed on substituted field"""
        return ufl.replace(self.energy, equilibrium)


    def _solve(self, alpha_old: dolfinx.fem.function.Function, state: dict, neig=None):
        """Compute derivatives and check positivity"""
        _u = state["u"]
        _alpha = state["alpha"]

        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction([self.V_alpha], restricted_dofs)

        if len(restricted_dofs[0])==0:
            # no damaging = elastic state
            _stability = True
            self.H = None
            self._H_csr = {"ai": np.array([]), "aj": np.array([]), "av": np.nan}

            return _stability
        else:
            u = Function(state["u"].function_space)
            equilibrium = {u: state["u"]}
            # damaging, compute Hessian
            _F = ufl.replace(self.F_[0], equilibrium)
            
            _P = self._potential_energy(equilibrium)
            F_ = ufl.derivative(
                        _P,
                        self.state[0],
                        ufl.TestFunction(state["alpha"].ufl_function_space()),
                    )

            H_ = ufl.algorithms.expand_derivatives(
                            ufl.derivative(
                                F_,
                                self.state[0],
                                ufl.TrialFunction(self.V_alpha),
                            )
                        )

            self.H_form = ufl.algorithms.expand_derivatives(
                            ufl.derivative(
                                _F,
                                self.state[0],
                                ufl.TrialFunction(self.V_alpha),
                            )
                        )
            Haa = dolfinx.fem.petsc.assemble_matrix(form(self.H_form), bcs=self.bcs)
            Haa.assemble()

            _Haa = dolfinx.fem.petsc.assemble_matrix(form(H_), bcs=self.bcs)
            _Haa.assemble()

            ai, aj, av = Haa.getValuesCSR()
            _stability = np.min(av) > 0
            # __import__('pdb').set_trace()

            logging.critical(f"-- Haa values = {av} --")
            logging.critical(f"-- _stability = {_stability} --")
            self.H = _Haa
            self._H_csr = {"ai": ai, "aj": aj, "av": av}

            return _stability

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
        with dolfinx.common.Timer(f"~Second Order: Cone Project"):
            
            # get the subvector associated to damage dofs with inactive constraints 
            _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
            _is = PETSc.IS().createGeneral(_dofs)
            _sub = v.getSubVector(_is)
            zero = _sub.duplicate()
            zero.zeroEntries()

            _sub.pointwiseMax(_sub, zero)
            v.restoreSubVector(_is, _sub)
        return


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
prefix = os.path.join(outdir, "test_NLB")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

def test_NLB(nest):

    with open("parameters.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = '1D'
    parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 100
    parameters["model"]["k_res"] = 1e-4
    parameters["model"]["k"] = 3
    parameters["model"]["N"] = 2
    parameters["loading"]["max"] = parameters["model"]["k"]
    parameters["loading"]["steps"] = 10

    parameters["geometry"]["geom_type"] = "discrete-damageable"
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]
    _N = parameters["model"]["N"]


    # Create the mesh of the specimen with given dimensions
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)


    outdir = "output"
    prefix = os.path.join(outdir, f"test_cone-N{parameters['model']['N']}")

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
    # u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    # u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], Lx)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_dofs_1 = locate_dofs_topological(V_u, mesh.topology.dim - 1, left_facets)
    left_dofs_2 = locate_dofs_topological(V_alpha, mesh.topology.dim - 1, left_facets)

    # right_dofs_ux = dolfinx.fem.locate_dofs_geometrical(
    #     (V_u.sub(0), V_u.sub(0).collapse()), lambda x: np.isclose(x[0], Lx)
    # )

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_dofs_1 = locate_dofs_topological(V_u, mesh.topology.dim - 1, right_facets)
    right_dofs_2 = locate_dofs_topological(V_alpha, mesh.topology.dim - 1, right_facets)

    bcs_u = [
        dolfinx.fem.dirichletbc(zero_u, left_dofs_1),
        dolfinx.fem.dirichletbc(u_, right_dofs_1),
    ]
    bcs_alpha = [
        dolfinx.fem.dirichletbc(zero_alpha, left_dofs_2),
        dolfinx.fem.dirichletbc(zero_alpha, right_dofs_2),
    ]

    bcs_z = bcs_u + bcs_alpha

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = (elastic_energy_density_atk(state) +
                    damage_energy_density(state)) * dx

    parameters.get("model")["k_res"] = 1e-04
    parameters.get("solvers").get("damage_elasticity")["alpha_tol"] = 1e-03
    parameters.get("solvers").get("damage")["type"] = "SNES"

    Eu = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
    Ealpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha))

    F = [Eu, Ealpha]
    z = [u, alpha]

    block_params = {}

    block_params["snes_type"] = "vinewtonrsls"
    block_params["snes_linesearch_type"] = "basic"
    block_params["snes_rtol"] = 1.0e-8
    block_params["snes_atol"] = 1.0e-8
    block_params["snes_max_it"] = 30
    block_params["snes_monitor"] = ""
    block_params["linesearch_damping"] = 0.5

    if nest:
        block_params["ksp_type"] = "cg"
        block_params["pc_type"] = "fieldsplit"
        block_params["fieldsplit_pc_type"] = "lu"
        block_params["ksp_rtol"] = 1.0e-10
    else:
        block_params["ksp_type"] = "preonly"
        block_params["pc_type"] = "lu"
        block_params["pc_factor_mat_solver_type"] = "mumps"

    parameters.get("solvers")['newton'] = block_params

    hybrid = HybridFractureSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)


    snes = hybrid.newton.snes

    lb = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)
    ub = dolfinx.fem.petsc.create_vector_nest(hybrid.newton.F_form)
    functions_to_vec([u_lb, alpha_lb], lb)
    functions_to_vec([u_ub, alpha_ub], ub)

    loads = [0.1, 1.0, 1.1]
    # loads = np.linspace(0.0, 1.3, 10)

    data = []
    
    norm_12_form = dolfinx.fem.form(
        (ufl.inner(alpha, alpha) + ufl.inner(ufl.grad(alpha), ufl.grad(alpha))) * dx)

    for i_t, t in enumerate(loads):

        # u_.interpolate(lambda x: (t * np.ones_like(x[0]), 0 * np.ones_like(x[1])))
        # u_.vector.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )

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

        logging.info(f"alpha vector norm: {alpha.vector.norm()}")
        logging.info(f"alpha lb norm: {alpha_lb.vector.norm()}")
        logging.info(f"alphadot norm: {alphadot.vector.norm()}")
        logging.info(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")

        rate_12_norm = np.sqrt(comm.allreduce(
            dolfinx.fem.assemble_scalar(
                hybrid.scaled_rate_norm(alpha, parameters))
                , op=MPI.SUM))
        
        logging.info(f"rate scaled alpha_12 norm: {rate_12_norm}")


        # dissipated_energy = comm.allreduce(
        #     dolfinx.fem.assemble_scalar(dolfinx.fem.form(model.damage_energy_density(state) * dx)),
        #     op=MPI.SUM,
        # )
        # elastic_energy = comm.allreduce(
        #     dolfinx.fem.assemble_scalar(dolfinx.fem.form(model.elastic_energy_density(state) * dx)),
        #     op=MPI.SUM,
        # )
        
        # We have the equilibrium displacement now
        # let's substitute in the total energy
        # to get P(\alpha) = E(u^*, \alpha) where u^* is the equilibrium displacement



        datai = {
            "it": i_t,
            "AM_F_alpha_H1": hybrid.data["error_alpha_H1"][-1],
            "AM_Fnorm": hybrid.data["error_residual_F"][-1],
            "NE_Fnorm": hybrid.newton.snes.getFunctionNorm(),
            "load" : t,
            "dissipated_energy" : dissipated_energy,
            "elastic_energy" : elastic_energy,
            "total_energy" : elastic_energy+dissipated_energy,
            "solver_data" : hybrid.data,
            "alphadot_norm": alphadot.vector.norm(),
            "rate_12_norm": rate_12_norm
            # "eigs" : stability.data["eigs"],
            # "stable" : stability.data["stable"],
            # "F" : _F
        }
        data.append(datai)


        # logging.info(f"getConvergedReason() {newton.snes.getConvergedReason()}")
        # logging.info(f"getFunctionNorm() {newton.snes.getFunctionNorm():.5e}")
        try:
            check_snes_convergence(hybrid.newton.snes)
        except ConvergenceError:
            logging.info("not converged")

        # assert newton.snes.getConvergedReason() > 0

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
        _plt.screenshot(f"{prefix}/test_NLB-{comm.size}-{i_t}.png")
        _plt.close()

    print(data)


    if comm.rank == 0:
        a_file = open(f"{prefix}/time_data.json", "w")
        json.dump(data, a_file)
        a_file.close()


def test_cone():
    """docstring for test_cone"""
        
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

    from dolfinx.io import XDMFFile, gmshio
    import logging
    from dolfinx.common import Timer, list_timings, TimingType

    """Discrete endommageable springs in series
            1         2        i        k
    0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
    u_0         u_1       u_2     u_i      u_k


    [WWW]: endommageable spring, alpha_i
    load: displacement hard-t

    """


    from solvers.function import functions_to_vec
    logging.getLogger().setLevel(logging.CRITICAL)

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


    with open("parameters.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # parameters["stability"]["cone"] = ""
    # parameters["cone"]["atol"] = 1e-7

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = '1D'
    parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 2
    parameters["model"]["k_res"] = 1e-4
    parameters["model"]["k"] = 4
    parameters["model"]["N"] = 3
    parameters["loading"]["max"] = parameters["model"]["k"]
    parameters["loading"]["steps"] = 30

    parameters["geometry"]["geom_type"] = "discrete-damageable"
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]
    _N = parameters["model"]["N"]


    # Create the mesh of the specimen with given dimensions
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

    outdir = "output"
    prefix = os.path.join(outdir, f"test_NLB")

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
        alpha = state["alpha"]
        u = state["u"]
        eps = ufl.grad(u)

        _mu = parameters["model"]['mu']
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


    stability = BifurcationSolver(
        total_energy, state, bcs, stability_parameters=parameters.get("stability")
    )


    cone = StabilitySolver(
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    nlb = NLBSolver(total_energy, state, bcs,
        nlb_parameters=parameters.get("stability"))


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
        "Hii": [],
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

        print("")
        print("")
        print("")
        print("")
        logging.critical(f"-- Solving for t = {t:3.2f} --")

        solver.solve()

        # n_eigenvalues = 10
        is_stable = stability.solve(alpha_lb)
        is_elastic = stability.is_elastic()
        inertia = stability.get_inertia()
        stable = cone._solve(alpha_lb)
        nlb_stable = nlb._solve(alpha_lb, state)



        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State is stable: {is_stable}")
        ColorPrint.print_bold(f"State is nlb-stable: {nlb_stable}")


        # postproc

        if nlb.H is not None:
            _Hii = nlb._H_csr["av"][0]
            _fig = plot_matrix(nlb.H)
            _fig.savefig(f"{prefix}/mat-Haa--{i_t}.png")
        else:
            _Hii = np.nan

        check_stability.append(is_stable)

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
        history_data["Hii"].append(_Hii)
        history_data["alpha_t"].append(state["alpha"].vector.array.tolist())
        history_data["u_t"].append(state["u"].vector.array.tolist())
        
        # logging.critical(f"u_t {u.vector.array}")
        # logging.critical(f"u_t norm {state['u'].vector.norm()}")

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



if __name__ == "__main__":
    # test_NLB(nest=False)
    test_cone()
