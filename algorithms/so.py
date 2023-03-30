import logging
from pydoc import cli
from time import clock_settime

from matplotlib.pyplot import cla
import dolfinx
from solvers import SNESSolver
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
from slepc4py import SLEPc
from dolfinx.cpp.log import log, LogLevel
import ufl
import numpy as np
from pathlib import Path
from dolfinx.io import XDMFFile, gmshio
import logging

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

from utils import norm_H1, norm_L2, ColorPrint
import sys
sys.path.append("../")
import solvers.restriction as restriction
import solvers.slepcblockproblem as eigenblockproblem

rank = comm.Get_rank()
size = comm.Get_size()


def info_dofmap(space, name=None):
    """Get information on the dofmap"""
    logging.info("\n")
    logging.info("rank", comm.rank, f"space {name}")

    dofmap = space.dofmap
    logging.info("rank", comm.rank, f"dofmap.bs {dofmap.bs}")
    logging.info(
        "rank",
        comm.rank,
        f"space.dofmap.dof_layout.num_dofs (per element) {space.dofmap.dof_layout.num_dofs}",
    )
    local_size = dofmap.index_map.size_local * dofmap.index_map_bs
    logging.info("rank", comm.rank, f"local_size {local_size}")

    logging.info(
        "rank",
        comm.rank,
        f"dofmap.index_map.size_global {dofmap.index_map.size_global}",
    )
    logging.info(
        "rank",
        comm.rank,
        f"dofmap.index_map.local_range {dofmap.index_map.local_range}",
    )
    logging.info(
        "rank",
        comm.rank,
        f"dofmap.index_map.global_indices {dofmap.index_map.global_indices()}",
    )
    logging.info(
        "rank", comm.rank, f"dofmap.index_map.num_ghosts {dofmap.index_map.num_ghosts}"
    )


class StabilitySolver:
    """Base class for stability analysis of a unilaterally constrained
    local minimisation problem of a given energy. Instantiates the
     form associated to the second derivative of the energy.
     Computes the inertia of the bilinear operator and solves the full
     eigenvalue problem."""

    def __init__(
        self,
        energy: ufl.form.Form,
        state: dict,
        bcs: list,
        nullspace=None,
        stability_parameters=None,
    ):
        self.state = [state["u"], state["alpha"]]
        alpha = self.state[1]
        self.parameters = stability_parameters

        self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space

        self.mesh = alpha.function_space.mesh

        L = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))
        self.lmbda0 = dolfinx.fem.Function(L)

        self.F_ = [
            ufl.derivative(
                energy, state["u"], ufl.TestFunction(state["u"].ufl_function_space())
            ),
            ufl.derivative(
                energy,
                state["alpha"],
                ufl.TestFunction(state["alpha"].ufl_function_space()),
            ),
        ]
        self.F = dolfinx.fem.form(self.F_)
        
        # Is the current state critical? 
        self._critical = False

        self.bcs = bcs["bcs_u"] + bcs["bcs_alpha"]
        pass

    def is_elastic(self) -> bool:
        """Returns whether or not the current state is elastic,
        based on the strict positivity of the gradient of E
        """
        etol = self.parameters.get("is_elastic_tol")
        E_alpha = dolfinx.fem.assemble_vector(self.F[1])

        coef = max(abs(E_alpha.array))
        coeff_glob = np.array(0.0, dtype=PETSc.ScalarType)

        comm.Allreduce(coef, coeff_glob, op=MPI.MAX)

        elastic = not np.isclose(coeff_glob, 0.0, atol=etol)

        return elastic

    def is_stable(self) -> bool:
        if self.is_elastic():
            return True
        else:
            raise NotImplementedError

    def get_inactive_dofset(self, a_old) -> set:
        """Computes the set of dofs where damage constraints are inactive
        based on the energy gradient and the ub constraint. The global
        set of inactive constraint-dofs is the union of constrained
        alpha-dofs and u-dofs.
        """
        gtol = self.parameters.get("inactiveset_gatol")
        pwtol = self.parameters.get("inactiveset_pwtol")
        V_u = self.state[0].function_space

        F = dolfinx.fem.petsc.assemble_vector(self.F[1])

        with F.localForm() as f_local:
            idx_grad_local = np.where(np.isclose(f_local[:], 0.0, atol=gtol))[0]

        with self.state[
            1
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

        if len(dofs_alpha_inactive) > 0:
            self._critical = True
        else:
            self._critical = False
        
        logging.critical(
            f"rank {comm.rank}) Current state is critical? {self._critical}"
        )
        if self._critical:
            logging.critical(
                f"rank {comm.rank})     > The cone is open 🍦"
            )

        # F.view()

        restricted_dofs = [dofs_u_all, dofs_alpha_inactive]

        localSize = F.getLocalSize()

        restricted = len(dofs_alpha_inactive)

        logging.debug(
            f"rank {comm.rank}) Restricted to (local) {restricted}/{localSize} nodes, {float(restricted/localSize):.1%} (local)",
        )

        return restricted_dofs

    def setup_eigensolver(self, eigen):
        eigen.eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eigen.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

        eigen.eps.setWhichEigenpairs(eigen.eps.Which.TARGET_REAL)

        st = eigen.eps.getST()
        st.setType("sinvert")
        st.setShift(-1.0e-3)

        eigen.eps.setTolerances(
            self.parameters["eigen"]["eig_rtol"], 
            self.parameters["eigen"]["eps_max_it"]
            )

        eigen.eps.setDimensions(self.parameters["maxmodes"], PETSc.DECIDE)
        eigen.eps.setFromOptions()
        # eigen.eps.view()

        return eigen

    def inertia_setup(self, constraints):

        # opts = PETSc.Options()
        # opts.prefixPush(prefix)

        # for k, v in self.parameters.get("inertia").items():
        #     print(f"{prefix}{k} {v}")
        #     opts[k] = v
        # opts.prefixPop()
        # pc.setFromOptions()
        pc = PETSc.PC().create(comm)
        prefix = "inertia"
        opts = PETSc.Options(prefix)
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "cholesky"
        opts["pc_factor_mat_solver_type"] = "mumps"
        opts["mat_mumps_icntl_24"] = 1
        opts["mat_mumps_icntl_13"] = 1

        opts_glob = PETSc.Options()
        opts_glob["mat_mumps_icntl_24"] = 1

        pc.setOptionsPrefix(prefix)
        pc.setFromOptions()

        H0 = [[None for i in range(2)] for j in range(2)]

        for i in range(2):
            for j in range(2):
                H0[i][j] = ufl.derivative(
                    self.F_[i],
                    self.state[j],
                    ufl.TrialFunction(self.state[j].function_space),
                )

        H_form = dolfinx.fem.form(H0)

        _H = dolfinx.fem.petsc.create_matrix_block(H_form)
        dolfinx.fem.petsc.assemble_matrix_block(_H, H_form, self.bcs)
        _H.assemble()
        rH = constraints.restrict_matrix(_H)
        # constraints.restrict_matrix(_H).copy(rH)

        pc.setOperators(rH)

        pc.setUp()
        self.inertia = pc

        return pc

    def get_inertia(self) -> (int, int, int):

        Fm = self.inertia.getFactorMatrix()
        (neg, zero, pos) = Fm.getInertia()

        return (neg, zero, pos)

    def normalise_eigen(self, u, mode="max-beta"):

        assert mode == "max-beta"
        v, beta = u[0], u[1]
        V_alpha_lrange = beta.function_space.dofmap.index_map.local_range

        coeff = max(abs(beta.vector[V_alpha_lrange[0] : V_alpha_lrange[1]]))
        coeff_glob = np.array(0.0, "d")

        comm.Allreduce(coeff, coeff_glob, op=MPI.MAX)

        logging.debug(f"{rank}, coeff_loc {coeff:.3f}")
        logging.debug(f"{rank}, coeff_glob {coeff_glob:.3f}")

        if coeff_glob == 0.0:
            log(
                LogLevel.INFO,
                "Damage eigenvector is null i.e. |β|={}".format(beta.vector.norm()),
            )
            return 0.0

        with v.vector.localForm() as v_local:
            v_local.scale(1.0 / coeff_glob)
        v.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        with beta.vector.localForm() as beta_local:
            beta_local.scale(1.0 / coeff_glob)
        beta.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        logging.debug(
            f"{rank}, beta range: ({min(beta.vector[V_alpha_lrange[0] : V_alpha_lrange[1]]):.3f},\
            {max(beta.vector[V_alpha_lrange[0] : V_alpha_lrange[1]]):.3f})"
        )
        return coeff_glob

    def postproc_eigs(self, eigs, eigen):
        pass

    def solve(self, alpha_old: dolfinx.fem.function.Function):

        self.data = {
            "stable": [],
            "neg_eigs": [],
            "zero_eigs": [],
            "pos_eigs": [],
            "elastic": [],
        }

        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction([self.V_u, self.V_alpha], restricted_dofs)

        self.inertia_setup(constraints)

        eigen = eigenblockproblem.SLEPcBlockProblemRestricted(
            self.F_,
            self.state,
            self.lmbda0,
            bcs=self.bcs,
            restriction=constraints,
            prefix="stability",
        )
        self.setup_eigensolver(eigen)

        # save an instance
        self.eigen = eigen

        eigen.solve()

        nev, ncv, mpd = eigen.eps.getDimensions()
        neig = self.parameters["maxmodes"]

        if neig is not None:
            neig_out = min(eigen.eps.getConverged(), neig)
        else:
            neig_out = eigen.eps.getConverged()

        log(LogLevel.INFO, f"Number of requested eigenvalues: {nev}")
        log(LogLevel.INFO, f"Number of requested column vectors: {ncv}")
        log(LogLevel.INFO, f"Number of mpd: {mpd}")
        log(LogLevel.INFO, f"converged {ncv:d}")
        # print(f"{rank}) mode {i}: {name} beta-norm {ur[1].vector.norm()}")

        # postprocess
        spectrum = []
        Kspectrum = []
        
        for i in range(neig_out):
            logging.debug(f"{rank}) Postprocessing mode {i}")
            v_n = dolfinx.fem.Function(self.V_u, name="Displacement perturbation")
            beta_n = dolfinx.fem.Function(self.V_alpha, name="Damage perturbation")
            eigval, ur, _ = eigen.getEigenpair(i)
            _ = self.normalise_eigen(ur)
            log(LogLevel.INFO, "")
            log(LogLevel.INFO, "i        k          ")
            log(LogLevel.INFO, "--------------------")
            log(LogLevel.INFO, "%d     %6e" % (i, eigval.real))

            with ur[0].vector.localForm() as v_loc, v_n.vector.localForm() as v_n_loc:
                v_loc.copy(result=v_n_loc)

            v_n.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            with ur[
                1
            ].vector.localForm() as b_loc, beta_n.vector.localForm() as b_n_loc:
                b_loc.copy(result=b_n_loc)

            beta_n.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            logging.debug(f"mode {i} {ur[0].name}-norm {ur[0].vector.norm()}")
            logging.debug(f"mode {i} {ur[1].name}-norm {ur[1].vector.norm()}")

            Kspectrum.append(
                {
                    "n": i,
                    "lambda": eigval.real,
                    "xk": ur
                }
            )

            spectrum.append(
                {
                    "n": i,
                    "lambda": eigval.real,
                    "v": v_n,
                    "beta": beta_n,
                }
            )

        spectrum.sort(key=lambda item: item.get("lambda"))
        unstable_spectrum = list(filter(lambda item: item.get("lambda") <= 0, spectrum))

        self.spectrum = unstable_spectrum
        self.Kspectrum = Kspectrum

        eigs = [mode["lambda"] for mode in spectrum]
        eig0, u0, _ = eigen.getEigenpair(0)

        self.minmode = u0
        self.mineig = eig0

        perturbations_v = [spectrum[i]["v"] for i in range(neig_out)]
        perturbations_beta = [spectrum[i]["beta"] for i in range(neig_out)]
        # based on eigenvalue
        stable = np.min(eigs) > float(self.parameters.get("eigen").get("eps_tol"))
        
        self.data = {
            "eigs": eigs,
            "perturbations_beta": perturbations_beta,
            "perturbations_v": perturbations_v,
            "stable": bool(stable),
        }

        return stable

    def save_eigenvectors(self, filename="output/eigvec.xdmf"):
        eigs = self.data["eigs"]
        v = self.data["perturbations_v"]
        beta = self.data["perturbations_beta"]
        ColorPrint.print_info("Saving the eigenvetors for the following eigenvalues")
        ColorPrint.print_info(eigs)

        if comm.rank == 0:
            out_dir = Path(filename).parent.absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, filename, "w") as ofile:
            ofile.write_mesh(self.mesh)
            for (i, eig) in enumerate(eigs):
                ofile.write_function(v[i], eig)
                ofile.write_function(beta[i], eig)


class ConeSolver(StabilitySolver):
    """Base class for a minimal implementation of the solution of eigenvalue
    problems bound to a cone. Based on numerical recipe SPA and KR result
    Thanks Yves and Luc."""
    def __init__(
        self,
        energy: ufl.form.Form,
        state: dict,
        bcs: list,
        nullspace=None,
        cone_parameters=None,
    ):    
        super(ConeSolver, self).__init__()

    def normalise_eigen(self, u, mode="norm"):
        assert mode == "norm"
        v, beta = u[0], u[1]
        V_alpha_lrange = beta.function_space.dofmap.index_map.local_range

        with v.vector.localForm() as v_local:
            v_local.scale(1.0 / coeff_glob)
        v.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        with beta.vector.localForm() as beta_local:
            beta_local.scale(1.0 / coeff_glob)
        beta.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        logging.debug(
            f"{rank}, beta range: ({min(beta.vector[V_alpha_lrange[0] : V_alpha_lrange[1]]):.3f},\
            {max(beta.vector[V_alpha_lrange[0] : V_alpha_lrange[1]]):.3f})"
        )
        return coeff_glob

    def solve(self, alpha_old: dolfinx.fem.function.Function, neig=None):
        # Loosely solve eigenproblem to get initial guess x_0
        # Project aka truncate u_k = x_0/phi(x_0)
        # compute residual, eigen_k, and eigenvect_k
        # update x_k
        x_diff = dolfinx.fem.Function(self.alpha.function_space)

        Kspectrum = []

        self.data = {
            "stable": [],
            "neg_eigs": [],
            "zero_eigs": [],
            "pos_eigs": [],
            "elastic": [],
        }

        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction([self.V_u, self.V_alpha], restricted_dofs)

        self.inertia_setup(constraints)

        eigen = eigenblockproblem.SLEPcBlockProblemRestricted(
            self.F_,
            self.state,
            self.lmbda0,
            bcs=self.bcs,
            restriction=constraints,
            prefix="cone",
        )
        self.setup_eigensolver(eigen)


        if neig is not None:
            eigen.eps.setDimensions(neig, PETSc.DECIDE)

        for iteration in range(
            self.solver_parameters.get("stability").get("cone").get("max_it")
        ):


            with dolfinx.common.Timer("~Cone Constrained : Internal iterations"):
                eigen.solve()

                nev, ncv, mpd = eigen.eps.getDimensions()
                if neig is not None:
                    neig_out = min(eigen.eps.getConverged(), neig)
                else:
                    neig_out = eigen.eps.getConverged()

                log(LogLevel.INFO, f"Number of requested eigenvalues: {nev}")
                log(LogLevel.INFO, f"Number of requested column vectors: {ncv}")
                log(LogLevel.INFO, f"Number of mpd: {mpd}")
                log(LogLevel.INFO, f"converged {ncv:d}")

                # xk.vector.copy(x_diff.vector)
                # x_diff.vector.axpy(-1, x_old.vector)
                # x_diff.vector.ghostUpdate(
                #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                # )

                logging.debug(f"{rank}) Postprocessing FIRST mode")
                xk = dolfinx.fem.Function(self.alpha.function_space)
                x_old = dolfinx.fem.Function(self.alpha.function_space)

                # v_n = dolfinx.fem.Function(self.V_u, name="Displacement perturbation")
                # beta_n = dolfinx.fem.Function(self.V_alpha, name="Damage perturbation")
                eigval, uk, _ = eigen.getEigenpair(i)
                _ = self.normalise_eigen(uk)

                with uk[1].vector.localForm() as beta_loc, xk.vector.localForm() as xk_loc:
                    beta_loc.copy(result=xk_loc)
                    xk_loc.vector.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                    )
                # compute lambdak
                # compute the residual
                # project onto cone  = componentwise truncation
                # 
                # 

                with xk.vector.localForm() as xk_loc, x_old.vector.localForm() as x_old_loc:
                    xk_loc.vector.copy(x_old_loc.vector)
                    x_old.vector.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                    )


            if (
                self.solver_parameters.get(
                    "stability").get("cone").get("criterion")
                == "standard"
            ):
                logging.info(
                    f"CO - Iteration: {iteration:3d}, Error: {norm_H1(x_diff):3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
                )
                # residual_y = assemble_vector(self.elasticity.F_form)

                if norm_H1(x_diff) <= self.solver_parameters.get(
                    "stability"
                ).get("cone").get("x_rtol"):
                    break
        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}"
            )

        for i in range(neig_out):
            logging.debug(f"{rank}) Postprocessing mode {i}")
            v_n = dolfinx.fem.Function(self.V_u, name="Displacement perturbation")
            beta_n = dolfinx.fem.Function(self.V_alpha, name="Damage perturbation")
            eigval, uk, _ = eigen.getEigenpair(i)
            _ = self.normalise_eigen(uk)
            log(LogLevel.INFO, "")
            log(LogLevel.INFO, "i        k          ")
            log(LogLevel.INFO, "--------------------")
            log(LogLevel.INFO, "%d     %6e" % (i, eigval.real))

            with uk[0].vector.localForm() as v_loc, v_n.vector.localForm() as v_n_loc:
                v_loc.copy(result=v_n_loc)

            v_n.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            with uk[1].vector.localForm() as b_loc, beta_n.vector.localForm() as b_n_loc:
                b_loc.copy(result=b_n_loc)

            beta_n.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            logging.debug(f"mode {i} {uk[0].name}-norm {uk[0].vector.norm()}")
            logging.debug(f"mode {i} {uk[1].name}-norm {uk[1].vector.norm()}")

            spectrum.append(
                {
                    "n": i,
                    "lambda": eigval.real,
                    "v": v_n,
                    "beta": beta_n,
                }
            )

        spectrum.sort(key=lambda item: item.get("lambda"))
        unstable_spectrum = list(filter(lambda item: item.get("lambda") <= 0, spectrum))

        self.spectrum = unstable_spectrum

        eigs = [mode["lambda"] for mode in spectrum]
        eig0, u0, _ = eigen.getEigenpair(0)

        self.minmode = u0
        self.mineig = eig0

        perturbations_v = [spectrum[i]["v"] for i in range(neig_out)]
        perturbations_beta = [spectrum[i]["beta"] for i in range(neig_out)]
        # based on eigenvalue
        stable = np.min(eigs) > float(self.parameters.get("eigen").get("eps_tol"))

        self.data = {
            "eigs": eigs,
            "perturbations_beta": perturbations_beta,
            "perturbations_v": perturbations_v,
            "stable": bool(stable),
        }

        return stable

    def save_eigenvectors(self, filename="output/eigvec.xdmf"):
        eigs = self.data["eigs"]
        v = self.data["perturbations_v"]
        beta = self.data["perturbations_beta"]
        ColorPrint.print_info("Saving the eigenvetors for the following eigenvalues")
        ColorPrint.print_info(eigs)

        if comm.rank == 0:
            out_dir = Path(filename).parent.absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, filename, "w") as ofile:
            ofile.write_mesh(self.mesh)
            for (i, eig) in enumerate(eigs):
                ofile.write_function(v[i], eig)
                ofile.write_function(beta[i], eig)


# more elegant:
    def Fs(self):
        """Nonlinear fixed point scheme, yet Lipschitz-continuous,
        given a vector x_k in K, returns x_{k+1}"""
        _lmbda = xAx/xBx
        # A.mult(e, y) # A*e = y
        # _y = Ax - _lmbda Bx
        # pick _s
        _u = self.xk - _s*_y
        return _PiK(_u)/phi(_u)
    
    def phi(self, v):
        """Normalisation function"""
        _coef = norm_H1(v)