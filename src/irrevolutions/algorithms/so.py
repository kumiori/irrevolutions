import logging
from pathlib import Path

import dolfinx
import numpy as np
import ufl
from dolfinx.cpp.la.petsc import get_local_vectors
from dolfinx.fem import Function
from dolfinx.fem.petsc import create_vector_block
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import irrevolutions.solvers.restriction as restriction
import irrevolutions.solvers.slepcblockproblem as eigenblockproblem
from irrevolutions.solvers.function import functions_to_vec, vec_to_functions
from irrevolutions.utils import ColorPrint, _logger, norm_L2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


class NonConvergenceException(Exception):
    def __init__(self, message="Non-convergence error"):
        """
        Exception class for non-convergence errors during computations.
        """
        self.message = message
        super().__init__(self.message)


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


class SecondOrderSolver:
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
        """
        Initialize the SecondOrderSolver.

        Args:
            energy (ufl.form.Form): The energy functional.
            state (dict): Dictionary containing state variables 'u' and 'alpha'.
            bcs (list): List of boundary conditions.
            nullspace: Nullspace object for the problem.
            stability_parameters: Parameters for the stability analysis.
        """
        self.state = [state["u"], state["alpha"]]
        alpha = self.state[1]
        self.parameters = stability_parameters

        # Initialize function spaces
        self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space

        self.mesh = alpha.function_space.mesh
        self.data = {}
        # Initialize L as a DG(0) function
        L = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))
        self.lmbda0 = dolfinx.fem.Function(L)

        # Define the forms associated with the second derivative of the energy
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

    def is_stable(self) -> bool:
        """
        Checks if the system is stable based on elasticity.
        FIXME: Implement stability check.

        Returns:
            bool: True if the system is stable, False otherwise.
        """
        if self.is_elastic():
            return True
        else:
            raise NotImplementedError("Stability check not implemented")

    def get_inactive_dofset(self, a_old) -> set:
        """Computes the set of dofs where damage constraints are inactive
        based on the energy gradient, the upper bound, and the lower bound
        constraint. The global set of inactive constraint-dofs is the union
        of constrained alpha-dofs and u-dofs.

        Computes the set of inactive dofs for damage constraints.

        Args:
            a_old: The old state vector.

        Returns:
            set: Set of inactive dofs.
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

        restricted_dofs = [dofs_u_all, dofs_alpha_inactive]

        localSize = F.getLocalSize()

        restricted = len(dofs_alpha_inactive)

        _logger.debug(
            f"rank {comm.rank}) Restricted to (local) {restricted}/{localSize} nodes, {float(restricted/localSize):.1%} (local)",
        )

        return restricted_dofs

    def setup_eigensolver(self, eigen):
        """
        Set up the eigenvalue solver for stability analysis.

        Args:
            eigen: Eigenvalue problem instance.

        Returns:
            eigen: Updated eigenvalue problem instance.
        """
        eigen.eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eigen.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

        eigen.eps.setWhichEigenpairs(eigen.eps.Which.TARGET_REAL)

        st = eigen.eps.getST()
        st.setType("sinvert")
        st.setShift(-1.0e-3)

        eigen.eps.setTolerances(
            self.parameters["eigen"]["eig_rtol"], self.parameters["eigen"]["eps_max_it"]
        )

        eigen.eps.setDimensions(self.parameters["maxmodes"], PETSc.DECIDE)
        eigen.eps.setFromOptions()
        # eigen.eps.view()

        return eigen

    def inertia_setup(self, constraints):
        """
        Set up the inertia matrix for the system.

        Args:
            constraints: Constraint object.

        Returns:
            pc: Preconditioner object.
        """
        pc = PETSc.PC().create(comm)
        prefix = "inertia_"
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
        """
        Get the inertia of the inertia matrix.

        Returns:
            Tuple[int, int, int]: Tuple containing the number of negative, zero, and positive eigenvalues.
        """
        Fm = self.inertia.getFactorMatrix()
        (neg, zero, pos) = Fm.getInertia()

        return (neg, zero, pos)

    def normalise_eigenmode(self, x, mode="functional"):
        """Normalises the eigenmode by the functional L2 norm

        Args:
            x (_type_): a (mixed space) vector
            mode (str, optional): _description_. Defaults to "functional".
        """
        _v = dolfinx.fem.Function(self.V_u, name="Displacement_component")
        _β = dolfinx.fem.Function(self.V_alpha, name="Damage_component")

        vec_to_functions(x, [_v, _β])

        if mode == "functional":
            scaling = np.sqrt(norm_L2(_v) ** 2 + norm_L2(_β) ** 2)
        else:
            raise NotImplementedError("Normalisation mode not implemented")

        # for u in [_v, _β]:
        #     with u.vector.localForm() as u_local:
        #         u_local.scale(1.0 / scaling)
        #     u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        with x.localForm() as x_local:
            x_local.scale(1.0 / scaling)
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        return x

    def normalise_eigen(self, u, mode="max-beta"):
        """
        Normalize the eigenmode vector.

        Args:
            u: Eigenmode vector.
            mode (str): Mode for normalization. Supported modes:
                - "max-beta", L-infty on beta
                - "unit", L2-norm of the mixed vector

        Returns:
            float: Coefficient used for normalization.
        """
        if mode == "max-beta":
            v, beta = u[0], u[1]
            coeff_glob = beta.vector.norm(3)

            logging.debug(f"{rank}, |β|_infty {beta.vector.norm(3):.3f}")

        elif mode == "unit":
            coeff_glob = np.sqrt(sum(n**2 for n in [v_i.vector.norm() for v_i in u]))
            logging.debug(f"rank {rank}, coeff_glob {coeff_glob:.3f}")
            logging.debug(f"{rank}, |(v, β)^*|_2 {coeff_glob:.3f}")

        if coeff_glob == 0.0:
            logging.error(f"Damage eigenvector is null i.e. |β|={beta.vector.norm()}")
            return 0.0

        for v_i in u:
            with v_i.vector.localForm() as v_local:
                v_local.scale(1.0 / coeff_glob)
            v_i.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

        _norm = np.sqrt(sum(n**2 for n in [v_i.vector.norm(2) for v_i in u]))

        return coeff_glob

    def postproc_eigs(self, eigs, eigen):
        """
        Postprocess the computed eigenvalues.

        Args:
            eigs: List of computed eigenvalues.
            eigen: Eigenvalue problem instance.
        """

    def solve(self, alpha_old: dolfinx.fem.function.Function):
        """
        Solve the stability analysis problem.

        Args:
            alpha_old: Old state vector.

        Returns:
            bool: True if stable, False otherwise.
        """
        # Initialize the data dictionary
        self.data = {
            "stable": [],
            "neg_eigs": [],
            "zero_eigs": [],
            "pos_eigs": [],
            "elastic": [],
        }
        self.alpha_old = alpha_old

        # Check if the system is damage-critical and log it
        self.log_critical_state()

        with dolfinx.common.Timer(f"~Second Order: Bifurcation") as timer:
            # Set up constraints
            constraints = self.setup_constraints(alpha_old)
            self.inertia_setup(constraints)

            # Set up and solve the eigenvalue problem
            eigen = self.setup_eigenvalue_problem(constraints)
            eigen.solve()

            # Process and analyze the eigenmodes
            self.eigen = eigen
            spectrum = self.process_eigenmodes(eigen)

            # Sort eigenmodes by eigenvalues
            spectrum.sort(key=lambda item: item.get("lambda"))
            # unstable_spectrum = list(filter(lambda item: item.get("lambda") <= 0, spectrum))
            
            # Store the results
            stable = self.store_results(eigen, spectrum)

        return stable

    def _is_critical(self, alpha_old):
        """
        Determines if the current state is damage-critical.

        Args:
            alpha_old (dolfinx.fem.function.Function): The previous damage function.

        Returns:
            bool: True if damage-critical, False otherwise.
        """
        constrained_dofs = len(self.get_inactive_dofset(alpha_old)[1])

        if constrained_dofs > 0:
            return True
        else:
            return False

    def log_critical_state(self):
        """Log whether the system is damage-critical."""
        critical = self._is_critical(self.alpha_old)
        _emoji = "💥" if critical else "🌪"

        _logger.info(
            f"rank {comm.rank}) Current state is damage-critical? {critical } {_emoji } "
        )
        _emoji = "non-trivial 🍦 (solid)" if critical else "trivial 🌂 (empty)"
        if critical:
            _logger.info(f"rank {comm.rank})         => The cone is {_emoji}")

    def setup_constraints(self, alpha_old: dolfinx.fem.function.Function):
        """Set up constraints and return them."""
        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction([self.V_u, self.V_alpha], restricted_dofs)
        return constraints

    def setup_eigenvalue_problem(self, constraints):
        """Set up the eigenvalue problem and return the solver."""
        eigen = eigenblockproblem.SLEPcBlockProblemRestricted(
            self.F_,
            self.state,
            self.lmbda0,
            bcs=self.bcs,
            restriction=constraints,
            prefix="stability",
        )
        self.setup_eigensolver(eigen)
        self.eigen = eigen  # Save the eigenvalue problem instance

        eigen.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(eigen.A, eigen.A_form, eigen.bcs)
        eigen.A.assemble()
        self.A_matrix = eigen.A

        return eigen

    def process_eigenmodes(self, eigen):
        """Process eigenmodes and return a list of unstable modes."""
        spectrum = []

        for i in range(self.get_number_of_process_eigenvalues(eigen)):
            logging.debug(f"{rank}) Postprocessing mode {i}")
            v_n, beta_n, eigval, _u = self.process_eigenmode(eigen, i)
            logging.debug("%d     %6e" % (i, eigval.real))
            spectrum.append(
                {
                    "n": i,
                    "lambda": eigval.real,
                    "xk": _u,
                    "v": v_n,
                    "beta": beta_n,
                }
            )

        return spectrum

    def get_number_of_process_eigenvalues(self, eigen):
        """Process a limited number of eigenvalues, limited by parameters or by
        the number of converged solutions"""

        neig = self.parameters["maxmodes"]

        if neig is not None:
            neig_out = min(eigen.eps.getConverged(), neig)
        else:
            neig_out = eigen.eps.getConverged()

        return neig_out

    def process_eigenmode(self, eigen, i):
        """Process a single eigenmode and return its components."""
        v_n = dolfinx.fem.Function(self.V_u, name="Displacement_perturbation")
        β_n = dolfinx.fem.Function(self.V_alpha, name="Damage_perturbation")
        _u = create_vector_block(self.F)
        eigval, ur, _ = eigen.getEigenpair(i)

        functions_to_vec(ur, _u)

        _u = self.normalise_eigenmode(_u, mode="functional")

        for u, component in zip(ur, [v_n, β_n]):
            with u.vector.localForm() as u_loc, component.vector.localForm() as c_loc:
                u_loc.copy(result=c_loc)
            component.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        return v_n, β_n, eigval, _u

    def store_results(self, eigen, spectrum):
        """Store eigenmodes and results."""

        if not spectrum:
            # Spectrum is empty, handle this case accordingly
            self.spectrum = []
            self._spectrum = []
            self.minmode = None
            self.mineig = None
            self.data = {
                "inf_spectrum": [],
                "eigs": [],
                "perturbations_beta": [],
                "perturbations_v": [],
                "stable": False,
            }
            self.perturbation = {}
            return False

        unstable_spectrum = list(filter(lambda item: item.get("lambda") <= 0, spectrum))

        # spectrum = unstable_spectrum

        self.spectrum = spectrum
        self._spectrum = unstable_spectrum

        eigs = [mode["lambda"] for mode in spectrum]

        # getEigenpair need not be ordered
        eig0, u0, _ = eigen.getEigenpair(spectrum[0]["n"])

        self.minmode = u0
        self.mineig = eig0

        perturbations_v = [mode["v"] for mode in unstable_spectrum]
        perturbations_beta = [mode["beta"] for mode in unstable_spectrum]

        stable = self.check_stability(eig0)

        self.data = {
            "inf_spectrum": self._spectrum,
            "eigs": eigs,
            "perturbations_beta": perturbations_beta,
            "perturbations_v": perturbations_v,
            "stable": bool(stable),
        }
        # store the first perturbation mode
        self.perturbation = {
            "v": spectrum[0]["v"],
            "β": spectrum[0]["beta"],
            "λ": spectrum[0]["lambda"],
        }

        return stable

    def check_stability(self, eig0):
        """Check stability based on eigenvalues and return the result."""
        eigs = [mode["lambda"] for mode in self.spectrum]
        # eig0, u0, _ = eigen.getEigenpair(0)

        if len(eigs) == 0:
            # assert
            stable = eig0.real > 0
        else:
            stable = np.min(eigs) > float(self.parameters.get("eigen").get("eps_tol"))

        return stable

    def save_eigenvectors(self, filename="output/eigvec.xdmf"):
        """
        Save computed eigenvectors to a file.

        Args:
            filename (str): Output filename for the XDMF file.
        """
        eigs = self.data["inf_spectrum"]
        v = self.data["perturbations_v"]
        beta = self.data["perturbations_beta"]
        ColorPrint.print_info("Saving the eigenvectors for the following eigenvalues")
        ColorPrint.print_info(eigs)

        if comm.rank == 0:
            out_dir = Path(filename).parent.absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, filename, "w") as ofile:
            ofile.write_mesh(self.mesh)
            for i, eig in enumerate(eigs):
                ofile.write_function(v[i], eig)
                ofile.write_function(beta[i], eig)


class BifurcationSolver(SecondOrderSolver):
    """Minimal implementation for the solution of the uniqueness issue"""

    def __init__(
        self,
        energy: ufl.form.Form,
        state: dict,
        bcs: list,
        nullspace=None,
        bifurcation_parameters=None,
    ):
        """
        Initialize the BifurcationSolver.

        Args:
            energy (ufl.form.Form): The energy functional.
            state (dict): Dictionary containing state variables 'u' and 'alpha'.
            bcs (list): List of boundary conditions.
            nullspace: Nullspace object for the problem.
            bifurcation_parameters: Parameters for the bifurcation analysis.
        """
        super(BifurcationSolver, self).__init__(
            energy,
            state,
            bcs,
            nullspace,
            stability_parameters=bifurcation_parameters,
        )

    def log(self, logger = logger):
        # Check if spectrum is available
        if not self._spectrum:
            logger.info("No negative spectrum.")

        # Find the minimum eigenvalue
        min_eigenvalue = min(entry["lambda"] for entry in self.spectrum if "lambda" in entry)
        # Determine if the evolution is unique (example condition)
        unique_evolution = all(entry["lambda"] > 0 for entry in self.spectrum)

        # Size of the computed spectrum
        spectrum_size = len(self.spectrum)

        # Log the information
        logger.info(f"Processed eigenvalues: {self.get_number_of_process_eigenvalues(self.eigen)}")
        logger.info(f"Inertia: {self.get_inertia()}")
        logger.info(f"Eigenvalues: {' '.join([f'{value:.1e}' for value in self.data['eigs']])}")
        
        logger.info(f"Minimum eigenvalue: {min_eigenvalue:.2f}")
        logger.info(f"Unique evolution: {unique_evolution}")
        logger.info(f"Size of computed spectrum: {spectrum_size}")

class StabilitySolver(SecondOrderSolver):
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
        """
        Initialize the StabilitySolver.

        Args:
            energy (ufl.form.Form): The energy functional.
            state (dict): Dictionary containing state variables 'u' and 'alpha'.
            bcs (list): List of boundary conditions.
            nullspace: Nullspace object for the problem.
            cone_parameters: Parameters for the  cone-stability analysis.
        """
        super(StabilitySolver, self).__init__(
            energy,
            state,
            bcs,
            nullspace,
            stability_parameters=cone_parameters,
        )

        self.solution = {"lambda_t": np.nan, "xt": [], "yt": []}

        with dolfinx.common.Timer(f"~Second Order: Stability"):
            with dolfinx.common.Timer(f"~Second Order: Cone Project"):
                # self._converged = False
                self._v = create_vector_block(self.F)

                self._reasons = {
                    "0": "converged",
                    "-1": "non-converged, check the logs",
                    "1": "converged atol",
                    "2": "converged residual",
                }
                _reason = None

    def solve(self, alpha_old: dolfinx.fem.function.Function, eig0=None, inertia=None):
        """
        Solves an abstract eigenvalue problem using the Scaling & Projection-Algorithm (SPA).

        Args:
            alpha_old (dolfinx.fem.function.Function): The previous damage function.
            eig0 (list): List of bifurcation eigenmodes, if available.
            inertia (list): Inertia of operator, if available.

        Returns:
            bool: True if the problem is stable, False if not.
        """

        self.sanity_check(eig0, inertia)
        self.iterations = 0

        # save an internal reference
        self.alpha_old = alpha_old

        self.data = {
            "error_x_L2": [],
            "lambda_k": [],
            "y_norm_L2": [],
        }
        self.solution = {"lambda_t": np.nan, "xt": None, "yt": None}

        if not self._is_critical(alpha_old):
            _logger.info(
                "the current state is damage-subcritical (hence elastic), the state is thus stable"
            )
            return True

        elif not eig0 and inertia[0] == 0 and inertia[1] == 0:
            _logger.info(
                "the current state is damage-critical and the evolution path is unique, the state is thus *Stable"
            )
            return True
        else:
            # assert len(eig0) > 0
            # assert that there is at least one negative or zero eigenvalue
            assert inertia[0] > 0 or inertia[1] > 0

            _x, _y, _Ax, self._xold = self.initialize_full_vectors()

            # x0 = eig0[0].get("xk")
            x0 = eig0
            x0.copy(result=_x).normalize()
            _x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            _logger.debug(f"initial guess x0: {x0.array}")
            self.x0 = x0.copy()

        errors = []

        self._converged = False
        errors.append(1)

        with dolfinx.common.Timer(f"~Second Order: Stability"):
            constraints = self.setup_constraints(alpha_old)
            self.constraints = constraints

            eigen = self.setup_eigenvalue_problem(constraints)

            self.eigen = eigen
            _Ar = constraints.restrict_matrix(eigen.A)
            _xk = constraints.restrict_vector(_x)
            _yr = constraints.restrict_vector(_y)
            self._Axr = constraints.restrict_vector(_Ax)
            self._xoldr = constraints.restrict_vector(self._xold)

            _lmbda_k = _xk.norm()

            self._residual_norm = 1.0

            self.Ar_matrix = _Ar.copy()
            _yr, _xk, _lmbda_k = self.convergence_loop(errors, _Ar, _xk)

            # process eigenmode
            # ... extend ...
            self._extend_vector(_yr, _y)
            self._extend_vector(_xk, _x)
            
            y = self.normalise_eigenmode(_y, mode="functional")
            xk = self.normalise_eigenmode(_x, mode="functional")

            # store
            self.store_results(_lmbda_k, xk, y)
            stable = self.check_stability(_lmbda_k)

        return stable

    def convergence_loop(self, errors, _Ar, _xk):
        """
        Perform a convergence loop to iteratively solve the variational inequality problem.

        This method iteratively updates the solution `_xk` until convergence is achieved
        based on the given convergence criteria `errors`.

        Parameters:
        - errors (list): List of error tolerances for convergence criteria.
        - _Ar (petsc4py.PETSc.Mat): Precomputed product of the system matrix `A` and the current solution `_xk`.
        - _xk (petsc4py.PETSc.Vec): Current solution vector.

        Returns:
        - _y (petsc4py.PETSc.Vec): Final solution vector.
        - _xk (petsc4py.PETSc.Vec): Updated solution vector after convergence.
        - _lmbda_k (float): Updated Lagrange multiplier corresponding to the final solution.
        """
        
        _s = float(self.parameters.get("cone").get("scaling"))

        while self.iterate(_xk, errors):
            _lmbda_k, _y = self.update_lambda_and_y(_xk, _Ar)
            _xk = self.update_xk(_xk, _y, _s)
            self.log_data(_xk, _lmbda_k, _y)
        return _y, _xk, _lmbda_k

    def update_lambda_and_y(self, xk, Ar):
        """
        Update the eigenvalue and solution vector based on the current solution `xk` and the product `Ar`.
            λ_k = <x_k, A x_k> / <x_k, x_k>
            y_k = A x_k - λ_k x_k

        Parameters:
        - xk (petsc4py.PETSc.Vec): Current solution vector.
        - Ar (petsc4py.PETSc.Mat): Precomputed product of the system matrix and the current solution.

        Returns:
        - _lmbda_t (float): Updated eigenvalue.
        - y (petsc4py.PETSc.Vec): Updated solution vector.
        """
        _Axr = xk.copy()
        y = xk.copy()

        Ar.mult(xk, _Axr)

        xAx_r = xk.dot(_Axr)

        _logger.debug(f"xk view in update at iteration {self.iterations}")

        _lmbda_t = xAx_r / xk.dot(xk)
        y.waxpy(-_lmbda_t, xk, _Axr)
        self._residual_norm = y.norm()

        return _lmbda_t, y

    def update_xk(self, xk, y, s):
        # Update _xk based on the scaling and projection algorithm
        xk.copy(result=self._xoldr)
        # x_k = x_k + (-s * y)

        xk.axpy(-s, y)

        _logger.debug(f"xk view before cone-project at iteration {self.iterations}")
        _cone_restricted = self._cone_project_restricted(xk)

        _logger.debug(f"xk view after cone-project at iteration {self.iterations}")
        n2 = _cone_restricted.normalize()

        # _logger.info(f"Cone project update: normalisation {n2}")

        return _cone_restricted

    def log_data(self, xk, lmbda_k, y):
        # Update SPA data during each iteration
        # self.iterations += 1
        # self.data["iterations"].append(self.iterations)
        self.data["lambda_k"].append(lmbda_k)
        self.data["y_norm_L2"].append(y.norm())
        # self.data["x_norm_L2"].append(xk.norm())
        self.data["error_x_L2"].append(self.error)

    def sanity_check(self, eig0, inertia):
        # this is done at each solve

        if not eig0 and inertia[0] == 0 and inertia[1] == 0:
            # no eigenvalues in the vector space and positive spectrum
            # the state is stable
            self.stable = True
            return self.stable

        self._aerrors = []
        self._rerrors = []

        return None

    def initialize_full_vectors(self):
        _x = create_vector_block(self.F)
        _y = create_vector_block(self.F)
        _Ax = create_vector_block(self.F)
        _xold = create_vector_block(self.F)

        return _x, _y, _Ax, _xold

    def initialize_restricted_vectors(self, constraints):
        # Create and initialize SPA vectors
        _x = create_vector_block(self.F)
        _y = create_vector_block(self.F)
        _Ax = create_vector_block(self.F)
        _xold = create_vector_block(self.F)

        _xk = constraints.restrict_vector(_x)
        _y = constraints.restrict_vector(_y)
        _xoldr = constraints.restrict_vector(_xold)
        _Axr = constraints.restrict_vector(_Ax)

        return _xk, _y, _xoldr, _Axr

    def finalise_eigenmode(self, xt, yt, lmbda_t):
        # Extract, extend, and finalize the converged eigenmode
        self._xk = xt

        (v, β) = (
            Function(self.V_u, name="Displacement_perturbation"),
            Function(self.V_alpha, name="Damage_perturbation"),
        )

        vec_to_functions(xt, [v, β])
        self.perturbation = {"v": v, "β": β, "λ": lmbda_t}

        self._y = create_vector_block(self.F)

        (w, ζ) = (
            Function(self.V_u, name="Displacement_residual"),
            Function(self.V_alpha, name="Damage_residual"),
        )

        vec_to_functions(yt, [w, ζ])
        self.residual = {"w": w, "ζ": ζ}

        return self.perturbation

    def iterate(self, x, errors):
        """
        Perform convergence check and handle exceptions (NonConvergenceException).

        Args:
            x: Current vector.
            errors: List to store errors.

        Returns:
            bool: True if converged, False otherwise.
        """
        try:
            converged = self._convergenceTest(x, errors)
        except NonConvergenceException as e:
            logging.warning(e)
            logging.warning("Continuing")
            converged = False
            # return False
        if not converged:
            self.iterations += 1
        else:
            self._converged = True

        # should we iterate?
        return False if converged else True

    def get_perturbation(self):
        """
        Get the perturbation vector.

        Returns:
            Union[dolfinx.fem.function.Function, None]: Perturbation vector if converged, None otherwise.
        """
        if self._converged:
            self._extend_vector(self.x_converged, self._v)
            return self._v
        else:
            return None

    def _convergenceTest(self, x, errors):
        """
        Test the convergence of the current iterate xk against the prior, restricted version.

        Args:
            x: Current iterate vector.
            errors: List to store errors.

        Returns:
            bool: True if converged, False otherwise.
        """

        _atol = self.parameters.get("cone").get("cone_atol")
        _rtol = self.parameters.get("cone").get("cone_rtol")
        _maxit = self.parameters.get("cone").get("cone_max_it")

        if self.iterations == _maxit:
            _reason = -1
            _logger.critical("Reached maxit without convergence")
            raise NonConvergenceException(
                f"SPA solver did not converge to atol {_atol} or rtol {_rtol} within maxit={_maxit} iterations."
            )

        diff = x.duplicate()
        diff.zeroEntries()
        # xdiff = x_old - x_k
        diff.waxpy(-1.0, self._xoldr, x)

        error_x_L2 = diff.norm()

        self.error = error_x_L2
        self._aerror = error_x_L2

        errors.append(error_x_L2)
        self._aerrors.append(self._aerror)

        if not self.iterations % 10000:
            _logger.critical(
                f"     [i={self.iterations}] error_x_L2 = {error_x_L2:.4e}, atol = {_atol}, res = {self._residual_norm}"
            )
            if self.iterations > 0:
                _logger.critical(
                f"     [i={self.iterations}] lambda_k = {self.data['lambda_k'].pop():.2e}, atol = {_atol}, res = {self._residual_norm}"
            )

        # self.data["iterations"].append(self.iterations)
        # self.data["error_x_L2"].append(error_x_L2)

        _acrit = self._aerror < self.parameters.get("cone").get("cone_atol")
        _rnorm = self._residual_norm < self.parameters.get("cone").get("cone_rtol")

        _crits = (_acrit, False)

        met_criteria = []

        for index, criterion in enumerate(_crits, start=1):
            if criterion:
                self._converged = True
                met_criteria.append(index)

        if len(met_criteria) >= 1:
            _reason = met_criteria
            _reason_str = [self._reasons[str(r)] for r in _reason]
            _logger.critical(
                f"     [i={self.iterations}] met criteria: {met_criteria}, reason(s) {_reason_str}"
            )
        # elif len(met_criteria) == 1:
        # _reason = met_criteria
        elif self.iterations == 0 or not met_criteria:
            self._converged = False
            _reason_str = "Not converged"

        return self._converged

    def _isin_cone(self, x):
        """
        Checks if the vector x is in the cone.

        Args:
            x: Vector to be checked.

        Returns:
            bool: True if x is in the cone, False otherwise.
        """
        if x.size != self._v.size:
            self._extend_vector(x, self._v)
            _x = self._v
        else:
            _x = x

        # Get the subvector associated with damage degrees of freedom with inactive constraints
        _dofs = self.constraints.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _sub = _x.getSubVector(_is)

        if not self.iterations:
            _logger.critical(
                f"ITER {self.iterations} rank {rank} is in the cone: {(_sub.array >= 0).all()}"
            )

        return (_sub.array >= 0).all()

    def _extend_vector(self, vres, vext):
        """
        Extends a restricted vector vr into v, not in place.

        Args:
            vres: Restricted vector to be extended.
            vext: Extended vector.

        Returns:
            None
        """

        vext.zeroEntries()

        vext.array[self.constraints.bglobal_dofs_vec_stacked] = vres.array

        return vext

    def _cone_project_restricted(self, v):
        """
        Projects a vector into the relevant cone, handling restrictions. Not in place.

        Args:
            v: Vector to be projected.

        Returns:
            Vector: The projected vector.
        """
        with dolfinx.common.Timer(f"~Second Order: Cone Project"):
            maps = [
                (V.dofmap.index_map, V.dofmap.index_map_bs)
                for V in self.constraints.function_spaces
            ]
            _x = create_vector_block(self.F)

            self._extend_vector(v, _x)

            # _logger.critical(f"rank {rank} viewing _x")
            # _x.view()

            with _x.localForm() as x_local:
                _dofs = self.constraints.bglobal_dofs_vec[1]
                x_local.array[_dofs] = np.maximum(x_local.array[_dofs], 0)

                _logger.debug(f"Local dofs: {_dofs}")
                _logger.debug(f"x_local")
                _logger.debug(f"x_local truncated")

            _x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            x_u, x_alpha = get_local_vectors(_x, maps)

            # _logger.info(f"Cone Project: Local data of the subvector x_u: {x_u}")
            # _logger.info(f"Cone Project: Local data of the subvector x_alpha: {x_alpha}")

            x = self.constraints.restrict_vector(_x)
            # __import__('pdb').set_trace()
            # _x.copy(result=x)
            _x.destroy()

        return x

    def check_stability(self, lmbda_t):
        # Check for stability based on SPA algorithm's convergence
        if self._converged and lmbda_t < float(
            self.parameters.get("cone").get("cone_rtol")
        ):
            return False
        elif self._converged and lmbda_t > float(
            self.parameters.get("cone").get("cone_rtol")
        ):
            return True

    def store_results(self, lmbda_t, xt, yt):
        # Store SPA results and log convergence information
        perturbation = self.finalise_eigenmode(xt, yt, lmbda_t)
        # self.data["lambda_0"] = lmbda_t
        self.solution = {"lambda_t": lmbda_t, "xt": xt, "yt": yt}
        self.perturbation = perturbation
        _logger.info(
            f"Convergence of SPA algorithm within {self.iterations} iterations"
        )
        _logger.info(f"Restricted Eigen _xk is in cone 🍦 ? {self._isin_cone(xt)}")

        _logger.critical(f"Restricted Eigenvalue {lmbda_t:.4e}")
        _logger.info(f"Restricted Eigenvalue is positive {lmbda_t > 0}")
        _logger.info(f"Restricted Error {self.error:.4e}")

    def log(self, logger=logger):
        # for key, value in self.data.items():
        #     logger.info(f"{key}: {value}")
        #  = {"lambda_t": lmbda_t, "xt": xt, "yt": yt}
        if self.solution['lambda_t'] is not np.nan:
            logger.info(f"Restricted Eigenvalue: {self.solution['lambda_t']}")
            logger.info(f"Restricted Eigenfunction is in cone 🍦 ? {self._isin_cone(self.solution['xt'])}")
            logger.info(f"Restricted Error {self.error:.4e}")

        return 
    
    def save_input_data(self, filename="data/input_data.xdmf"):
        """
        Save input data to a file.

        Args:
            filename (str): Output filename for the XDMF file.
        """
        if comm.rank == 0:
            out_dir = Path(filename).parent.absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, filename, "w") as ofile:
            ofile.write_mesh(self.mesh)
            ofile.write_function(self.alpha_old, 0.0)

        # Use a try/except block to handle the case when A_matrix is not available
        try:
            from os import path

            import test_binarydataio as bio

            # Save data if available
            if hasattr(self, "A_matrix") and self.A_matrix is not None:
                bio.save_binary_data(path.join(out_dir, "A_hessian.mat"), self.A_matrix)
                bio.save_binary_data(
                    path.join(out_dir, "Ar_hessian.mat"), self.Ar_matrix
                )
            else:
                _logger.warning(
                    "Warning: A_matrix is not available. Skipping its save."
                )

            if hasattr(self, "x0") and self.x0 is not None:
                bio.save_binary_data(path.join(out_dir, "x0.vec"), self.x0)
            else:
                _logger.warning(
                    "Warning: x0_vector is not available. Skipping its save."
                )

            if hasattr(self, "constraints") and self.constraints is not None:
                bio.save_minimal_constraints(
                    self.constraints, path.join(out_dir, "constraints.pkl")
                )
            else:
                _logger.warning(
                    "Warning: x0_vector is not available. Skipping its save."
                )

            # Save minimal constraints

        except Exception as e:
            _logger.error(f"Error during data save: {str(e)}")
