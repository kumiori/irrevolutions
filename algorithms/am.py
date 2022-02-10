import logging
import dolfinx
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
from petsc4py import PETSc
import ufl
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD

from utils import norm_H1, norm_L2

class AlternateMinimisation:
    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
        monitor=None,
    ):
        self.u = state["u"]
        self.alpha = state["alpha"]
        # self.bcs  = bcs
        self.alpha_old = Function(self.alpha.function_space)
        self.alpha.vector.copy(self.alpha_old.vector)
        self.alpha.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.total_energy = total_energy

        self.state = state
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]
        self.total_energy = total_energy
        # self.solver_parameters = default_parameters["solvers"]
        # if solver_parameters:
        self.solver_parameters = solver_parameters

        self.monitor = monitor

        # self.dx = ufl.Measure("dx", self.alpha.function_space.mesh)

        V_u = self.u.function_space
        V_alpha = self.alpha.function_space

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
        # Set near nullspace for the gamg preconditioner for elasticity

        # if np.not_equal(V_u.mesh.geometry.dim, 1):
        #     null_space = build_nullspace_elasticity(V_u)
        #     self.elasticity.a.setNearNullSpace(null_space)

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
            "error_residual_u": [],
            "solver_alpha_reason": [],
            "solver_alpha_it": [],
            "solver_u_reason": [],
            "solver_u_it": [],
            "total_energy": [],
        }
        if outdir:
            with XDMFFile(
                comm,
                f"{outdir}/fields.xdmf",
                "w",
                encoding=XDMFFile.Encoding.HDF5,
            ) as file:
                file.write_mesh(self.u.function_space.mesh)

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

            Fv = [dolfinx.fem.assemble_vector(form(F)) for F in self.F]

            Fnorm = np.sqrt(
                np.array(
                    [comm.allreduce(np.linalg.norm(Fvi) ** 2, op=MPI.SUM)
                     for Fvi in Fv]
                ).sum()
            )

            error_alpha_max = alpha_diff.vector.max()[1]
            total_energy_int = comm.allreduce(
                dolfinx.fem.assemble_scalar(form(self.total_energy)), op=MPI.SUM
            )
            residual_u = dolfinx.fem.assemble_vector(self.elasticity.F_form)
            residual_u.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            dolfinx.fem.set_bc(residual_u, self.elasticity.bcs, self.u.vector)
            error_residual_u = ufl.sqrt(residual_u.dot(residual_u))

            self.alpha.vector.copy(self.alpha_old.vector)
            self.alpha_old.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            self.data["iteration"].append(iteration)
            self.data["error_alpha_L2"].append(error_alpha_L2)
            self.data["error_alpha_H1"].append(error_alpha_H1)
            self.data["F_norm"].append(Fnorm)
            self.data["error_alpha_max"].append(error_alpha_max)
            self.data["error_residual_u"].append(error_residual_u)
            self.data["solver_alpha_it"].append(solver_alpha_it)
            self.data["solver_alpha_reason"].append(solver_alpha_reason)
            self.data["solver_u_reason"].append(solver_u_reason)
            self.data["solver_u_it"].append(solver_u_it)
            self.data["total_energy"].append(total_energy_int)

            if outdir:
                with XDMFFile(
                    comm,
                    f"{outdir}/fields.xdmf",
                    "a",
                    encoding=XDMFFile.Encoding.HDF5,
                ) as file:
                    file.write_function(self.u, iteration)
                    file.write_function(self.alpha, iteration)

            if self.monitor is not None:
                self.monitor(self)

            if (
                self.solver_parameters.get(
                    "damage_elasticity").get("criterion")
                == "residual_u"
            ):
                logging.info(
                    f"AM - Iteration: {iteration:3d}, Error: {error_residual_u:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
                )
                if error_residual_u <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
            if (
                self.solver_parameters.get(
                    "damage_elasticity").get("criterion")
                == "alpha_H1"
            ):
                logging.info(
                    f"AM - Iteration: {iteration:3d}, Error: {error_alpha_H1:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
                )
                if error_alpha_H1 <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}"
            )
