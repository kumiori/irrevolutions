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
from solvers.function import functions_to_vec

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
        logging.critical(f'is_elastic coeff_glob = {coeff_glob}')
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
            f"rank {comm.rank}) Current state is damage-critical? {self._critical}"
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

    def _solve(self, alpha_old: dolfinx.fem.function.Function, x0=None):
        """Recursively solves (until convergence) the abstract eigenproblem
        K \ni x \perp y := Ax - \lambda B x \in K^*
        based on the SPA recipe, cf. ...
        """
        _s = float(self.parameters.get("cone").get("scaling"))
        self.iterations = 0
        errors = []
        self.stable = True
        stable = self.solve(alpha_old)
        self.data = {
            "iterations": [],
            "error_x_L2": [],
            "lambda_k": [],
            "lambda_0": [],
            "y_norm_L2": [],
        }
        
        self._converged = False

        # The cone is non-trivial, aka non-empty
        # only if the state is irreversibly damage-critical

        if self._critical:
            errors.append(1)
            self.data["y_norm_L2"].append(1)

            # loop

            _x = dolfinx.fem.petsc.create_vector_block(self.F)        
            _y = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Ax = dolfinx.fem.petsc.create_vector_block(self.F)        
            _Bx = dolfinx.fem.petsc.create_vector_block(self.F)        
            self._xold = dolfinx.fem.petsc.create_vector_block(self.F)    
            
            # Map current solution into vector _x
            if x0 is None:
                functions_to_vec(self.Kspectrum[0].get("xk"), _x)
            else:
                functions_to_vec(x0, _x)

            logging.critical(f"         Size of _x : {_x.size}")
            self.data["lambda_k"].append(self.Kspectrum[0].get("lambda"))
            logging.critical(f'         initial lambda-guess : {self.Kspectrum[0].get("lambda")}')
            if not self.eigen.empty_B():
                logging.debug("B = Id")

            with dolfinx.common.Timer(f"~Second Order: Cone Solver - SPA s={_s}"):
                assert self._xold.size == _x.size
        
                while not self.converged(_x):
                    errors.append(self.error)

                    # make it admissible: map into the cone
                    # logging.critical(f"_x is in the cone? {self._isin_cone(_x)}")

                    # self._cone_project(_x)
                    _x = self._cone_project_restricted(_x)

                    # logging.critical(f"_x is in the cone? {self._isin_cone(_x)}")
                    # K_t spectrum:
                    # compute {lambdat, xt, yt}

                    if self.eigen.restriction is not None:
                        _A = self.eigen.rA
                        _B = self.eigen.rB

                        # _x = self.eigen.restriction.restrict_vector(_x)
                        _y = self.eigen.restriction.restrict_vector(_y)
                        _Ax = self.eigen.restriction.restrict_vector(_Ax)
                        _Bx = self.eigen.restriction.restrict_vector(_Bx)

                        _xold = self.eigen.restriction.restrict_vector(self._xold)
                    else:
                        _A = self.eigen.A
                        _B = self.eigen.B
                        _xold = self._xold

                    _A.mult(_x, _Ax)
                    xAx = _x.dot(_Ax)

                    # compute: lmbda_t
                    if not self.eigen.empty_B():
                        _B.mult(_x, _Bx)
                        xBx = _x.dot(_Bx)
                        _lmbda_t = xAx/xBx
                    else:
                        _Bx = _x
                        _lmbda_t = xAx / _x.dot(_x)

                    # compute: y_t = _Ax - _lmbda_t * _Bx
                    _y.waxpy(-_lmbda_t, _Bx, _Ax)

                    # construct perturbation
                    # _v = _x - _s*y_t
        
                    # _x.copy(self._xold)
                    _x.copy(_xold)
                    _x.axpy(-_s, _y)
                    
                    # project onto cone
                    # self._cone_project(_x)
                    _x = self._cone_project_restricted(_x)

                    # assert self._xold.size == _x.size
                    
                    _xold.copy(self._xold)
                    
                    # L2-normalise
                    n2 = _x.normalize()
                    # _x.view()
                    # iterate
                    # x_i+1 = _v 

                    self.data["lambda_k"].append(_lmbda_t)
                    self.data["y_norm_L2"].append(_y.norm())




            self.data["iterations"] = self.iterations
            self.data["error_x_L2"] = errors
            self.data["lambda_0"] = _lmbda_t

            logging.critical(f"Convergence of SPA algorithm with s={_s} in {self.iterations} iterations")
            logging.critical(f"Eigenfunction is in cone? {self._isin_cone(_x)}")
            logging.critical(f"Eigenvalue {_lmbda_t}")

            # if (self._isin_cone(_x)):
                # bifurcating out of existence, not out of a numerical test
            if (self._converged and _lmbda_t < float(self.parameters.get("cone").get("cone_rtol"))):
                stable = bool(False)
            else:
                stable = bool(True)
        return bool(stable)

    def _is_critical(self, alpha_old):
        """is this a damage-critical state?"""
        constrained_dofs = len(self.get_inactive_dofset(alpha_old)[1])

        if constrained_dofs > 0:
            return bool(True)
        else:
            return bool(False)


    def my_solve(self, alpha_old: dolfinx.fem.function.Function, x0=None):
        _s = float(self.parameters.get("cone").get("scaling"))
        self.iterations = 0
        errors = []
        self.stable = True
        self.data = {
            "iterations": [],
            "error_x_L2": [],
            "lambda_k": [],
            "lambda_0": [],
            "y_norm_L2": [],
        }
        # Map current solution into vector _x
        _x = dolfinx.fem.petsc.create_vector_block(self.F)        
        if x0 is None:
            stable = self.solve(alpha_old)
            functions_to_vec(self.Kspectrum[0].get("xk"), _x)
        else:
            functions_to_vec(x0, _x)
        # __import__('pdb').set_trace()

        if not self._is_critical(alpha_old):
            return bool(True)

        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction([self.V_u, self.V_alpha], restricted_dofs)

        self._converged = False
        errors.append(1)
        self.data["y_norm_L2"].append(1)

        # initialise forms, matrices, vectors
        eigen = eigenblockproblem.SLEPcBlockProblemRestricted(
            self.F_,
            self.state,
            self.lmbda0,
            bcs=self.bcs,
            restriction=constraints,
            prefix="stability",
        )

        eigen.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(eigen.A, eigen.A_form, eigen.bcs)
        eigen.A.assemble()
        _Ar = constraints.restrict_matrix(eigen.A)

        _y = dolfinx.fem.petsc.create_vector_block(self.F)        
        _Ax = dolfinx.fem.petsc.create_vector_block(self.F)        
        
        self._xold = dolfinx.fem.petsc.create_vector_block(self.F)        
         
        _xk = constraints.restrict_vector(_x)
        self._xoldr = constraints.restrict_vector(self._xold)
    
        # _x = constraints.restrict_vector(_x)
        _y = constraints.restrict_vector(_y)
        _Axr = constraints.restrict_vector(_Ax)
        # _Ar = constraints.restrict_matrix(_A)

        with dolfinx.common.Timer(f"~Second Order: Cone Solver - SPA s={_s}"):
            while not self.converged(_xk):
                errors.append(self.error)

                _Ar.mult(_xk, _Axr)
                xAx_r = _xk.dot(_Axr)
                # B=id for us
                _lmbda_t = xAx_r / _xk.dot(_xk)
                _y.waxpy(-_lmbda_t, _xk, _Axr)
                
                _xk.copy(self._xoldr)
                _xk.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                # update current iterate
                _xk.axpy(-_s, _y)
                # normalise eigen
                n2 = _xk.normalize()

                self.data["lambda_k"].append(_lmbda_t)
                self.data["y_norm_L2"].append(_y.norm())

        self.data["iterations"] = self.iterations
        self.data["error_x_L2"] = errors
        self.data["lambda_0"] = _lmbda_t

        logging.critical(f"Convergence of SPA algorithm with s={_s} in {self.iterations} iterations")
        # logging.critical(f"Eigenfunction is in cone? {self._isin_cone(_xk)}")
        logging.critical(f"Errors {errors}")
        logging.critical(f"Eigenvalue {_lmbda_t}")

        # if (self._isin_cone(_x)):
        # bifurcating out of existence, not out of a numerical test

        if (self._converged and _lmbda_t < float(self.parameters.get("cone").get("cone_rtol"))):
            stable = bool(False)
        else:
            stable = bool(True)
    
        return bool(stable)

    def converged(self, x):
        converged = self._convergenceTest(x)
        
        if not converged:
            self.iterations += 1

        return converged


    def _convergenceTest(self, x):
        """Test convergence of current iterate xk against 
        prior, restricted version"""
        # _atol = self.parameters.get("eigen").get("eps_tol")
        # _maxit = self.parameters.get("eigen").get("eps_max_it")

        _atol = self.parameters.get("cone").get("cone_atol")
        _maxit = self.parameters.get("cone").get("cone_max_it")

        if self.iterations == _maxit:
            raise RuntimeError(f'SPA solver did not converge within {_maxit} iterations. Aborting')
            # return False        
        diff = x.duplicate()
        diff.zeroEntries()

        # xdiff = -x + x_old
        diff.waxpy(-1., self._xoldr, x)
        error_x_L2 = diff.norm()

        self.error = error_x_L2
        if not self.iterations % 100:
            logging.critical(f"     [i={self.iterations}] error_x_L2 = {error_x_L2}")

        self.data["iterations"].append(self.iterations)
        self.data["error_x_L2"].append(error_x_L2)

        if error_x_L2 < _atol:
            self._converged = True
        elif self.iterations == 0 or error_x_L2 >= _atol:
            self._converged = False

        return self._converged

    # def converged(self, x):
    #     converged = self.convergenceTest(x)
        
    #     # update xold
    #     # x.copy(self._xold)
    #     # x.vector.ghostUpdate(
    #     #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    #     # )

    #     if not converged:
    #         self.iterations += 1

    #     return converged

    def convergenceTest(self, x):
        """Test convergence of current iterate x against 
        prior"""
        # _atol = self.parameters.get("eigen").get("eps_tol")
        # _maxit = self.parameters.get("eigen").get("eps_max_it")

        _atol = self.parameters.get("cone").get("cone_atol")
        _maxit = self.parameters.get("cone").get("cone_max_it")

        if self.eigen.restriction is not None:
            _x = self.eigen.restriction.restrict_vector(x)
            _xold = self.eigen.restriction.restrict_vector(self._xold)
        else:
            _x = x
            _xold = self._xold

        if self.iterations == _maxit:
            raise RuntimeError(f'SPA solver did not converge within {_maxit} iterations. Aborting')
            # return False        
        diff = _x.duplicate()
        diff.zeroEntries()

        # xdiff = -x + x_old
        diff.waxpy(-1., _xold, _x)
        error_x_L2 = diff.norm()

        self.error = error_x_L2
        if not self.iterations % 100:
            logging.critical(f"     [i={self.iterations}] error_x_L2 = {error_x_L2}")

        self.data["iterations"].append(self.iterations)
        self.data["error_x_L2"].append(error_x_L2)

        if error_x_L2 < _atol:
            self._converged = True
        elif self.iterations == 0 or error_x_L2 >= _atol:
            self._converged = False

        return self._converged

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
            # logging.critical(f"num dofs {len(self.eigen.restriction.bglobal_dofs_vec[1])}")
            # get the subvector associated to damage dofs with inactive constraints 
            _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
            _is = PETSc.IS().createGeneral(_dofs)
            _sub = v.getSubVector(_is)
            zero = _sub.duplicate()
            zero.zeroEntries()

            _sub.pointwiseMax(_sub, zero)
            v.restoreSubVector(_is, _sub)
        return

    def _extend_vector(self, vres, vext):
        """extends restricted vector vr into v, in place"""
        # v = dolfinx.fem.petsc.create_vector_block(F)

        _isall = PETSc.IS().createGeneral(self.eigen.restriction.bglobal_dofs_vec_stacked)
        _suball = vext.getSubVector(_isall)

        vres.copy(_suball)
        vext.restoreSubVector(_isall, _suball)
        
        return

        
    def _cone_project_restricted(self, v):
        """Projects vector into the relevant cone
            handling restrictions.

            takes arguments:
            - v: vector in a mixed space

            returns
        """
        with dolfinx.common.Timer(f"~Second Order: Cone Project"):
            # logging.critical(f"num dofs {len(self.eigen.restriction.bglobal_dofs_vec[1])}")
            # get the subvector associated to damage dofs with inactive constraints 

            
            # if is already restricted
            if v.size == len(self.eigen.restriction.bglobal_dofs_vec_stacked):
                # use all dofs
                _dofs = np.arange(len(self.eigen.restriction.blocal_dofs[1])).tolist()
            # else get from restriction
            else:
                _dofs = self.eigen.restriction.bglobal_dofs_vec[1]

            _is = PETSc.IS().createGeneral(_dofs)
            logging.debug(f"rank {comm.rank}) IS.size from block-local dofs {_is.size}")
            logging.debug(f"rank {comm.rank}) v size                          {v.size}")
            logging.debug(f"rank {comm.rank}) IS Indices from block-local dofs {_is.getIndices()}")
            logging.debug(f"rank {comm.rank}) Restricted dofs {len(self.eigen.restriction.blocal_dofs[1])}")
            
            _sub = v.getSubVector(_is)
            zero = _sub.duplicate()

            zero.zeroEntries()

            _sub.pointwiseMax(_sub, zero)
            v.restoreSubVector(_is, _sub)

            if self.eigen.restriction is not None and v.size != len(self.eigen.restriction.bglobal_dofs_vec_stacked):
                return self.eigen.restriction.restrict_vector(v)
            else:
                return v
        return


    def _cone_restrict_project(self, v):
        """returns the projection of a full state vector v
        (considering the restriction), onto the positive cone
        the returned vector (new) is defined on the same space as v"""

        vk = v.copy()
        zero = v.duplicate()
        zero.zeroEntries()

        _is = PETSc.IS().createGeneral(self.eigen.restriction.bglobal_dofs_vec[1])
        _sub = vk.getSubVector(_is)
        print(f"{rank}) _sub-.array_r {_sub.array_r}")
        _subzero = zero.getSubVector(_is)
        _sub.pointwiseMax(_sub, _subzero)
        print(f"{rank}) _sub+.array_r {_sub.array_r}")
        vk.restoreSubVector(_is, _sub)

        return vk
