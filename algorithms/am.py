from utils import norm_H1, norm_L2
import logging
import dolfinx
from solvers import SNESSolver
from solvers.snesblockproblem import SNESBlockProblem
from solvers.function import functions_to_vec
from utils import set_vector_to_constant

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    assemble_scalar,
    locate_dofs_geometrical,
)
from petsc4py import PETSc
import ufl
import numpy as np
from dolfinx.io import XDMFFile

from mpi4py import MPI
comm = MPI.COMM_WORLD
try:
    from dolfinx.fem import (
        assemble_matrix,
        apply_lifting,
        create_vector,
        create_matrix,
        set_bc,
        assemble_vector
    )

except ImportError:
    from dolfinx.fem.petsc import (
        assemble_matrix,
        apply_lifting,
        create_vector,
        create_matrix,
        set_bc,
        assemble_vector
    )
logging.basicConfig()

logging.getLogger().setLevel(logging.INFO)


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
            "error_residual_F": [],
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

        for iteration in range(1,
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
            residual_u = assemble_vector(self.elasticity.F_form)
            residual_u.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(residual_u, self.elasticity.bcs, self.u.vector)
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
            self.data["error_residual_F"].append(Fnorm)
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
                logging.critical(
                    f"AM - Iteration: {iteration:3d}, Error:  ||Du E||_L2 {error_residual_u:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
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
                logging.critical(
                    f"AM - Iteration: {iteration:3d}, Error ||Δα_i||_H1: {error_alpha_H1:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
                )
                if error_alpha_H1 <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}"
            )


import solvers.restriction as restriction

class AwareMinimisation(AlternateMinimisation):
    """Alternate Minimisation, aware of exit strategy"""
    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
        monitor=None,
    ):
        self.bcs = bcs
        super(AwareMinimisation, self).__init__(
            total_energy, state, bcs, solver_parameters, bounds, monitor
        )


        self.F_ = [
            ufl.derivative(
                total_energy, state["u"], ufl.TestFunction(state["u"].ufl_function_space())
            ),
            ufl.derivative(
                total_energy,
                state["alpha"],
                ufl.TestFunction(state["alpha"].ufl_function_space()),
            ),
        ]
        self.Fform = dolfinx.fem.form(self.F_)

        Jform = [
            [None for i in range(len(state))] for j in range(len(state))
        ]

        for (i, iv) in enumerate(self.state):
            for (j, jv) in enumerate(self.state):
                print(f'd_{jv} F_{i}')
                Jform[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        self.F_[i],
                        self.state[jv],
                        ufl.TrialFunction(self.state[jv].function_space),
                    )
                )

                # If the form happens to be empty replace with None
                if Jform[i][j].empty():
                    Jform[i][j] = None
        # __import__('pdb').set_trace()
        Jform = [[None, None], [None, None]]
        self.Jform = dolfinx.fem.form(Jform)

    def get_inactive_dofset(self, a_old) -> set:
        """Computes the set of dofs where damage constraints are inactive
        based on the energy gradient and the ub constraint. The global
        set of inactive constraint-dofs is the union of constrained
        alpha-dofs and u-dofs.
        """
        gtol = self.solver_parameters.get('hybrid').get("inactiveset_gatol")
        pwtol = self.solver_parameters.get('hybrid').get("inactiveset_pwtol")
        V_u = self.state['u'].function_space

        F = dolfinx.fem.petsc.assemble_vector(self.Fform[1])

        with F.localForm() as f_local:
            idx_grad_local = np.where(np.isclose(f_local[:], 0.0, atol=gtol))[0]

        with self.state[
            'alpha'
        ].vector.localForm() as a_local, a_old.vector.localForm() as a_old_local:
            idx_ub_local = np.where(np.isclose(a_local[:], 1.0, rtol=pwtol))[0]
            idx_lb_local = np.where(np.isclose(a_local[:], a_old_local[:], rtol=pwtol))[
                0
            ]

        idx_alpha_local = set(idx_grad_local).difference(
            set().union(idx_ub_local, idx_lb_local)
        )

        V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)

        dofs_u_all = np.arange(V_u_size, dtype=np.int32)
        dofs_alpha_inactive = np.array(list(idx_alpha_local), dtype=np.int32)

        restricted_dofs = [dofs_u_all, dofs_alpha_inactive]

        localSize = F.getLocalSize()

        restricted = len(dofs_alpha_inactive)

        logging.debug(
            f"rank {comm.rank}) Restricted to (local) {restricted}/{localSize} nodes, {float(restricted/localSize):.1%} (local)",
        )

        return restricted_dofs


    def solve(self, outdir=None):
        super(AwareMinimisation, self).solve(outdir)
        # V_u = self.state['u'].function_space
        # V_alpha = self.state['alpha'].function_space

        # restricted_dofs = self.get_inactive_dofset(self.alpha_old)
        # constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)
                
        # Fv = dolfinx.fem.petsc.create_vector_block(self.Fform)
        # Frx = constraints.restrict_vector(Fv)
        # # __import__('pdb').set_trace()

        # dolfinx.fem.petsc.assemble_vector_block(
        #     Frx,
        #     self.Fform,
        #     self.Jform,
        #     self.bcs['bcs_u'] + self.bcs['bcs_alpha']
        # )

        # Frx.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )

        # error_residual_Frx = Frx.norm(2)

        # if (
        #     self.solver_parameters.get(
        #         "damage_elasticity").get("criterion")
        #     == "residual_u"
        # ):
        # logging.critical(
        #     f"AM - Iteration: xxx, Error:  ||Frx||_L2 {error_residual_Frx:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
        # )


class HybridFractureSolver(AlternateMinimisation):
    """Hybrid (AltMin+Newton) solver for fracture"""

    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
        monitor=None,
    ):
        super(HybridFractureSolver, self).__init__(
            total_energy, state, bcs, solver_parameters, bounds, monitor
        )

        self.u_lb = dolfinx.fem.Function(state['u'].function_space, name="displacement lower bound")
        self.u_ub = dolfinx.fem.Function(state['u'].function_space, name="displacement upper bound")
        self.alpha_lb = dolfinx.fem.Function(state['alpha'].function_space, name="damage lower bound")
        self.alpha_ub = dolfinx.fem.Function(state['alpha'].function_space, name="damage upper bound")

        set_vector_to_constant(self.u_lb.vector, PETSc.NINFINITY)
        set_vector_to_constant(self.u_ub.vector, PETSc.PINFINITY)
        set_vector_to_constant(self.alpha_lb.vector, 0)
        set_vector_to_constant(self.alpha_ub.vector, 1)

        self.z = [self.u, self.alpha]
        bcs_z = bcs.get("bcs_u") + bcs.get("bcs_alpha")
        self.prefix = "blocknewton"

        nest = False
        self.newton = SNESBlockProblem(
            self.F, self.z, bcs=bcs_z, nest=nest, prefix="block"
        )
        newton_options = self.solver_parameters.get("newton", self.default_options())
        self.set_newton_options(newton_options)

        self.lb = dolfinx.fem.petsc.create_vector_nest(self.newton.F_form)
        self.ub = dolfinx.fem.petsc.create_vector_nest(self.newton.F_form)
        functions_to_vec([self.u_lb, self.alpha_lb], self.lb)
        functions_to_vec([self.u_ub, self.alpha_ub], self.ub)

    def default_options(self):
        opts = PETSc.Options(self.prefix)

        opts.setValue("snes_type", "vinewtonrsls")
        opts.setValue("snes_linesearch_type", "basic")
        opts.setValue("snes_rtol", 1.0e-08)
        opts.setValue("snes_atol", 1.0e-08)
        opts.setValue("snes_max_it", 30)
        opts.setValue("snes_monitor", "")
        opts.setValue("linesearch_damping", 0.5)

        nest = False

        if nest:
            opts.setValue("ksp_type", "cg")
            opts.setValue("pc_type", "fieldsplit")
            opts.setValue("fieldsplit_pc_type", "lu")
            opts.setValue("ksp_rtol", 1.0e-10)
        else:
            opts.setValue("ksp_type", "preonly")
            opts.setValue("pc_type", "lu")
            opts.setValue("pc_factor_mat_solver_type", "mumps")

        return opts

    def set_newton_options(self, newton_options):

        opts = PETSc.Options(self.prefix)
        # opts.prefixPush(self.prefix)
        logging.info(newton_options)

        for k, v in newton_options.items():
            opts[k] = v

        # opts.prefixPop()
        self.newton.snes.setOptionsPrefix(self.prefix)
        self.newton.snes.setFromOptions()

    def compute_bounds(self, v, alpha_lb):
        __import__('pdb').set_trace()
        lb = dolfinx.fem.create_vector_nest(v)
        ub = dolfinx.fem.create_vector_nest(v)

        with lb.getNestSubVecs()[0].localForm() as u_sub:
            u_sub.set(PETSc.NINFINITY)

        with ub.getNestSubVecs()[0].localForm() as u_sub:
            u_sub.set(PETSc.PINFINITY)

        with lb.getNestSubVecs()[
            1
        ].localForm() as alpha_sub, alpha_lb.vector.localForm() as alpha_lb_loc:
            alpha_lb_loc.copy(result=alpha_sub)

        with ub.getNestSubVecs()[1].localForm() as alpha_sub:
            alpha_sub.set(1.0)

        return lb, ub

    def getReducedNorm(self):
        """Retrieve reduced residual"""
        self.newton.compute_norms_block(self.newton.snes)
        return self.newton.norm_r

        # self.newton
        

    def solve(self, outdir=None):
        # Perform AM as customary
        super().solve(outdir)
        self.newton_data = {
            "iteration": [],
            "residual_Fnorm": [],
            "residual_Frxnorm": []
        }
        # update bounds and perform Newton step
        # lb, ub = self.compute_bounds(self.newton.F_form, self.alpha)
        functions_to_vec([self.u_lb, self.alpha_lb], self.lb)

        self.newton.snes.setVariableBounds(self.lb, self.ub)
        
        self.newton.solve(u_init=[self.u, self.alpha])

        self.newton_data["iteration"].append(self.newton.snes.getIterationNumber() + 1)
        self.newton_data["residual_Fnorm"].append(self.newton.snes.getFunctionNorm())
        self.newton_data["residual_Frxnorm"].append(self.getReducedNorm())

        self.data.update(self.newton_data)

        # self.data.append(newton_data)
        # self.data["newton_Fnorm"].append(Fnorm)