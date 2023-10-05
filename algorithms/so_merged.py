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
from solvers.function import vec_to_functions

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

        # Initialize L as a DG(0) function
        L = dolfinx.fem.FunctionSpace(self.mesh, ("DG", 0))
        self.lmbda0 = dolfinx.fem.Function(L)

        # Define the forms associated with the second derivative of the energy
        self.F_ = [
            ufl.derivative(
                energy, state["u"], ufl.TestFunction(
                    state["u"].ufl_function_space())
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

        Checks if the current state is elastic based on the gradient of E.

        Returns:
            bool: True if the state is elastic, False otherwise.
        """
        etol = self.parameters.get("is_elastic_tol")
        E_alpha = dolfinx.fem.assemble_vector(self.F[1])

        coef = max(abs(E_alpha.array))
        coeff_glob = np.array(0.0, dtype=PETSc.ScalarType)

        comm.Allreduce(coef, coeff_glob, op=MPI.MAX)

        elastic = np.isclose(coeff_glob, 0.0, atol=etol)
        logging.debug(f'is_elastic coeff_glob = {coeff_glob}')
        return elastic

    def is_stable(self) -> bool:
        """
        Checks if the system is stable based on elasticity.

        Returns:
            bool: True if the system is stable, False otherwise.
        """
        if self.is_elastic():
            return True
        else:
            raise NotImplementedError("Stability check not implemented")

    def get_inactive_dofset(self, a_old) -> set:
        """Computes the set of dofs where damage constraints are inactive
        based on the energy gradient and the ub constraint. The global
        set of inactive constraint-dofs is the union of constrained
        alpha-dofs and u-dofs.

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
            idx_grad_local = np.where(
                np.isclose(f_local[:], 0.0, atol=gtol))[0]

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

        _emoji = "💥" if self._critical else "🌪"
        logging.debug(
            f"rank {comm.rank}) Current state is damage-critical? {self._critical } {_emoji } "
        )

        _emoji = "non-trivial 🍦 (solid)" if self._critical else "trivial 🌂 (empty)"
        if self._critical:
            logging.debug(
                f"rank {comm.rank})     > The cone is {_emoji}"
            )

        restricted_dofs = [dofs_u_all, dofs_alpha_inactive]

        localSize = F.getLocalSize()

        restricted = len(dofs_alpha_inactive)

        logging.debug(
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
            self.parameters["eigen"]["eig_rtol"],
            self.parameters["eigen"]["eps_max_it"]
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
        """
        Get the inertia of the inertia matrix.

        Returns:
            Tuple[int, int, int]: Tuple containing the number of negative, zero, and positive eigenvalues.
        """
        Fm = self.inertia.getFactorMatrix()
        (neg, zero, pos) = Fm.getInertia()

        return (neg, zero, pos)

    def normalise_eigen(self, u, mode="max-beta"):
        """
        Normalize the eigenmode vector.

        Args:
            u: Eigenmode vector.
            mode (str): Mode for normalization. Only "max-beta" is supported.
        
        Returns:
            float: Coefficient used for normalization.
        """
        assert mode == "max-beta"
        v, beta = u[0], u[1]
        V_alpha_lrange = beta.function_space.dofmap.index_map.local_range

        coeff = max(abs(beta.vector[V_alpha_lrange[0]: V_alpha_lrange[1]]))
        coeff_glob = np.array(0.0, "d")

        comm.Allreduce(coeff, coeff_glob, op=MPI.MAX)

        logging.debug(f"{rank}, coeff_loc {coeff:.3f}")
        logging.debug(f"{rank}, coeff_glob {coeff_glob:.3f}")

        if coeff_glob == 0.0:
            log(
                LogLevel.INFO,
                "Damage eigenvector is null i.e. |β|={}".format(
                    beta.vector.norm()),
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
        """
        Postprocess the computed eigenvalues.

        Args:
            eigs: List of computed eigenvalues.
            eigen: Eigenvalue problem instance.
        """
        pass

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
            spectrum = self.process_eigenmodes(eigen)

            # Sort eigenmodes by eigenvalues
            spectrum.sort(key=lambda item: item.get("lambda"))
            unstable_spectrum = list(filter(lambda item: item.get("lambda") <= 0, spectrum))

            # Store the results
            stable = self.store_results(eigen, unstable_spectrum)

        return stable

    def log_critical_state(self):
        """Log whether the system is damage-critical."""
        _emoji = "💥" if self._critical else "🌪"
        logging.critical(
            f"rank {comm.rank}) Current state is damage-critical? {self._critical } {_emoji } "
        )
        _emoji = "non-trivial 🍦 (solid)" if self._critical else "trivial 🌂 (empty)"
        if self._critical:
            logging.debug(
                f"rank {comm.rank})     > The cone is {_emoji}"
            )

    def setup_constraints(self, alpha_old: dolfinx.fem.function.Function):
        """Set up constraints and return them."""
        restricted_dofs = self.get_inactive_dofset(alpha_old)
        constraints = restriction.Restriction(
            [self.V_u, self.V_alpha], restricted_dofs)
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
        return eigen

    def process_eigenmodes(self, eigen):
        """Process eigenmodes and return a list of unstable modes."""
        spectrum = []

        for i in range(self.get_number_of_process_eigenvalues(eigen)):
            logging.debug(f"{rank}) Postprocessing mode {i}")
            v_n, beta_n, eigval, _u = self.process_eigenmode(eigen, i)
            logging.critical("%d     %6e" % (i, eigval.real))
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
        v_n = dolfinx.fem.Function(self.V_u, name="Displacement perturbation")
        beta_n = dolfinx.fem.Function(self.V_alpha, name="Damage perturbation")

        _u = dolfinx.fem.petsc.create_vector_block(self.F)
        eigval, ur, _ = eigen.getEigenpair(i)
        _ = self.normalise_eigen(ur)

        functions_to_vec(ur, _u)

        with ur[0].vector.localForm() as v_loc, v_n.vector.localForm() as v_n_loc:
            v_loc.copy(result=v_n_loc)

        v_n.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with ur[1].vector.localForm() as b_loc, beta_n.vector.localForm() as b_n_loc:
            b_loc.copy(result=b_n_loc)

        beta_n.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.debug(f"mode {i} {ur[0].name}-norm {ur[0].vector.norm()}")
        logging.debug(f"mode {i} {ur[1].name}-norm {ur[1].vector.norm()}")
        logging.debug(f"mode {i} beta_n-norm {beta_n.vector.norm()}")
        logging.debug("")

        return v_n, beta_n, eigval, _u

    def store_results(self, eigen, unstable_spectrum):
        """Store eigenmodes and results."""
        spectrum = unstable_spectrum

        self.spectrum = spectrum
        self._spectrum = spectrum

        eigs = [mode["lambda"] for mode in spectrum]
        eig0, u0, _ = eigen.getEigenpair(0)

        self.minmode = u0
        self.mineig = eig0

        perturbations_v = [mode["v"] for mode in unstable_spectrum]
        perturbations_beta = [mode["beta"] for mode in unstable_spectrum]

        stable = self.check_stability(eig0)

        self.data = {
            "eigs": eigs,
            "perturbations_beta": perturbations_beta,
            "perturbations_v": perturbations_v,
            "stable": bool(stable),
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
            stable = np.min(eigs) > float(
                self.parameters.get("eigen").get("eps_tol"))

        return stable

    def save_eigenvectors(self, filename="output/eigvec.xdmf"):
        """
        Save computed eigenvectors to a file.

        Args:
            filename (str): Output filename for the XDMF file.
        """
        eigs = self.data["eigs"]
        v = self.data["perturbations_v"]
        beta = self.data["perturbations_beta"]
        ColorPrint.print_info(
            "Saving the eigenvectors for the following eigenvalues")
        ColorPrint.print_info(eigs)

        if comm.rank == 0:
            out_dir = Path(filename).parent.absolute()
            out_dir.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, filename, "w") as ofile:
            ofile.write_mesh(self.mesh)
            for (i, eig) in enumerate(eigs):
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
            cone_parameters: Parameters for the stability analysis with cones.
        """
        super(StabilitySolver, self).__init__(
            energy,
            state,
            bcs,
            nullspace,
            stability_parameters=cone_parameters,
        )
        self._converged = False
        self._v = dolfinx.fem.petsc.create_vector_block(self.F)

        self._reasons = {'0': 'converged',
                         '-1': 'non-converged, check the logs',
                         '1': 'converged atol',
                         '2': 'converged rtol'
                         }
        self._reason = None
        self.errors = []

        self.data = {
            "iterations": [],
            "error_x_L2": [],
            "lambda_k": [],
            "lambda_0": [],
            "y_norm_L2": [],
        }

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

    def my_solve(self, alpha_old: dolfinx.fem.function.Function, eig0=None):
        """
        Solves an abstract eigenvalue problem using the Scaling & Projection-Algorithm (SPA).

        Args:
            alpha_old (dolfinx.fem.function.Function): The previous damage function.
            eig0 (list): List of eigenmodes, if available.

        Returns:
            bool: True if the problem is stable, False if not.
        """

        stable = False
        
        # Initialize SPA parameters and data
        self.initialize_spa(alpha_old, eig0)

        # Check for stability using pre-computed eigenmodes if available
        if eig0 is None:
            stable = self.solve(alpha_old)
            eig0 = self._spectrum

        # Check for early termination conditions
        if len(eig0) == 0 or stable:
            return bool(stable)

        with dolfinx.common.Timer(f"~Second Order: Stability"):

            constraints = self.setup_constraints(alpha_old)
            eigen = self.setup_eigenvalue_problem(constraints)

            # Initialize SPA vectors and matrices
            _x, _y, _Ax, _Ar, _xk = self.initialize_vectors(constraints)
            # __import__('pdb').set_trace()
            _xk = self.initialize_xk(eig0, constraints)

            _y, _lmbda_t = self.initialize_y_lambda(_xk, _Ar, _y)

            # Apply SPA iterations
            while self.iterate(_xk, self.errors):
                _lmbda_t, _y = self.update_lambda_and_y(_xk, _Ar, _y)
                self.update_xk(_xk, _y, self._s)
                self.update_data(_xk, _lmbda_t, _y)

            # Extract and save the converged SPA eigenmode
            self.finalize_eigenmode(_xk)

            # Store SPA results
            self.store_results(_lmbda_t)

            # Check for stability based on the SPA algorithm's convergence
            stable = self.check_stability(_lmbda_t)

        return stable


    def initialize_spa(self, alpha_old: dolfinx.fem.function.Function, eig0=None):
        # Initialization of SPA parameters and data
        self.iterations = 0
        self.stable = True

        _s = float(self.parameters.get("cone").get("scaling"))
        self._s = _s
        self.errors = []
        self._rerrors = []
        self._aerrors = []

        if eig0 is None:
            self.stable = self.solve(alpha_old)
            eig0 = self._spectrum

        if len(eig0) == 0 or self.stable:
            return self.stable

        x0 = eig0[0].get("xk")
        _x = x0.copy()
        self.data["iterations"].append(0)
        self.data["error_x_L2"].append(1)

        return None


    def initialize_vectors(self, constraints):
        # Create and initialize SPA vectors 
        _x = dolfinx.fem.petsc.create_vector_block(self.F)
        _y = dolfinx.fem.petsc.create_vector_block(self.F)
        _Ax = dolfinx.fem.petsc.create_vector_block(self.F)
        self._xold = dolfinx.fem.petsc.create_vector_block(self.F)

        dolfinx.fem.petsc.assemble_matrix_block(self.eigen.A, self.eigen.A_form, self.eigen.bcs)

        self.eigen.A.assemble()

        _Ar = constraints.restrict_matrix(self.eigen.A)

        _xk = constraints.restrict_vector(_x)
        self._xoldr = constraints.restrict_vector(self._xold)

        _y = constraints.restrict_vector(_y)
        self._Axr = constraints.restrict_vector(_Ax)



        return _x, _y, _Ax, _Ar, _xk


    def initialize_xk(self, eig0, constraints):
        # Initialize the SPA iterate vector _xk
        x0 = eig0[0].get("xk")
        _x = x0.copy()
        _xk = constraints.restrict_vector(_x)

        self._xk = _xk
        # self._extend_vector(_xk, self._v)
        return _xk


    def initialize_y_lambda(self, xk, _Ar, _y):
        # Initialize SPA vectors _y and _Ax and calculate _lmbda_t
        
        # Axr = A * xk |_{restricted}

        _Ar.mult(xk, self._Axr)

        xAx_r = xk.dot(self._Axr)

        _lmbda_t = xAx_r / xk.dot(xk)
        _y.waxpy(-_lmbda_t, xk, self._Axr)

        return _y, _lmbda_t


    # def iterate(self, xk):
    #     # Perform one SPA iteration and return whether to continue
    #     self.update_lambda_and_y(xk, _Axr, _y)
    #     self.update_xk(xk, _y)
    #     self.update_data(xk, _lmbda_t, _y)

    #     return not self._converged


    def update_lambda_and_y(self, xk, Ar, y):
        # Update _lmbda_t and _y based on _xk, _Axr, and y
        Ar.mult(xk, self._Axr)
        
        xAx_r = xk.dot(self._Axr)

        _lmbda_t = xAx_r / xk.dot(xk)
        y.waxpy(-_lmbda_t, xk, self._Axr)

        return _lmbda_t, y


    def update_xk(self, xk, y, s):
        # Update _xk based on the scaling and projection algorithm
        xk.copy(self._xoldr)
        self._xoldr.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        xk.axpy(-s, y)

        self._cone_project_restricted(xk)
        n2 = xk.normalize()


    def update_data(self, xk, lmbda_t, y):
        # Update SPA data during each iteration
        self.iterations += 1
        self.data["iterations"].append(self.iterations)
        self.data["lambda_k"].append(lmbda_t)
        self.data["y_norm_L2"].append(y.norm())


    def finalize_eigenmode(self, xk):
        # Extract and finalize the converged SPA eigenmode
        self._xk = xk
        self._extend_vector(xk, self._v)

        (v, β) = (Function(self.V_u, name="Displacement perturbation"),
                Function(self.V_alpha, name="Damage perturbation"))

        vec_to_functions(self._v, [v, β])
        self.perturbation = {"v": v, "beta": β}


    def store_results(self, lmbda_t):
        # Store SPA results and log convergence information
        self.data["error_x_L2"].append(self.error)
        self.data["lambda_0"].append(lmbda_t)
        logging.info(
            f"Convergence of SPA algorithm within {self.iterations} iterations")
        logging.info(
            f"Restricted Eigen _xk is in cone 🍦 ? {self._isin_cone(self._xk)}")
        logging.critical(f"Restricted Eigenvalue {lmbda_t:.4e}")
        logging.critical(f"Restricted Eigenvalue is positive {lmbda_t > 0}")
        logging.info(f"Restricted Error {self.error:.4e}")
        logging.critical(
            f"Eigenfunction is in cone? {self._isin_cone(self._v)}")


    def check_stability(self, lmbda_t):
        # Check for stability based on SPA algorithm's convergence
        if self._converged and lmbda_t < float(self.parameters.get("cone").get("cone_rtol")):
            return False
        else:
            return True

    def iterate(self, x, errors):
        """
        Perform convergence check and handle exceptions (NonConvergenceException).

        Args:
            x: Current vector.
            errors: List to store errors.

        Returns:
            bool: True if converged, False otherwise.
        """
        converged = False
        try:
            converged = self._convergenceTest(x, errors)
        except NonConvergenceException as e:
            logging.warning(e)
            logging.warning("Continuing")
            # return False

        if not converged:
            self.iterations += 1
        else:
            self._converged = True
            self.x_converged = x.copy()

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
        assert x.norm() > 0, "Norm of x is zero"

        _atol = self.parameters.get("cone").get("cone_atol")
        _rtol = self.parameters.get("cone").get("cone_rtol")
        _maxit = self.parameters.get("cone").get("cone_max_it")

        if self.iterations == _maxit:
            self._reason = -1
            raise NonConvergenceException(
                f'SPA solver did not converge to atol {_atol} or rtol {_rtol} within maxit={_maxit} iterations.')

        diff = x.duplicate()
        diff.zeroEntries()

        # xdiff = -x + x_old
        diff.waxpy(-1., self._xoldr, x)

        error_x_L2 = diff.norm()

        self.error = error_x_L2
        self._aerror = error_x_L2
        self._rerror = error_x_L2 / x.norm()

        errors.append(error_x_L2)
        self._aerrors.append(self._aerror)
        self._rerrors.append(self._rerror)

        if not self.iterations % 1000:
            logging.critical(
                f"     [i={self.iterations}] error_x_L2 = {error_x_L2:.4e}, atol = {_atol}")
            # logging.critical(f"     [i={self.iterations}] x norm = {x.norm():.4e}, atol = {_atol}")

        self.data["iterations"].append(self.iterations)
        self.data["error_x_L2"].append(error_x_L2)

        _acrit = self._aerror < self.parameters.get("cone").get("cone_atol")
        _rcrit = self._rerror < self.parameters.get("cone").get("cone_rtol")

        _crits = (_acrit, _rcrit)

        met_criteria = []

        for index, criterion in enumerate(_crits, start=1):
            if criterion:
                self._converged = True
                met_criteria.append(index)

        if len(met_criteria) > 1:
            self._reason = 0
        elif len(met_criteria) == 1:
            self._reason = met_criteria
        elif self.iterations == 0 or not met_criteria:
            self._converged = False

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
        _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _sub = _x.getSubVector(_is)

        return (_sub.array >= 0).all()

    def _extend_vector(self, vres, vext):
        """
        Extends a restricted vector vr into v, in place.

        Args:
            vres: Restricted vector to be extended.
            vext: Extended vector.

        Returns:
            None
        """
        _isall = PETSc.IS().createGeneral(self.eigen.restriction.bglobal_dofs_vec_stacked)
        _suball = vext.getSubVector(_isall)

        vres.copy(_suball)
        vext.restoreSubVector(_isall, _suball)

    def _cone_project_restricted(self, v):
        """
        Projects a vector into the relevant cone, handling restrictions. In place.

        Args:
            v: Vector to be projected.

        Returns:
            Vector: The projected vector.
        """
        with dolfinx.common.Timer(f"~Second Order: Cone Project"):
            # get the subvector associated with damage degrees of freedom with inactive constraints

            if v.size != self._v.size:
                self._extend_vector(v, self._v)
                _v = self._v
            else:
                _v = v

            _dofs = self.eigen.restriction.bglobal_dofs_vec[1]
            _is = PETSc.IS().createGeneral(_dofs)

            _sub = _v.getSubVector(_is)
            zero = _sub.duplicate()

            zero.zeroEntries()

            _sub.pointwiseMax(_sub, zero)
            _v.restoreSubVector(_is, _sub)

            # if self.eigen.restriction is not None and v.size != len(self.eigen.restriction.bglobal_dofs_vec_stacked):
            if self.eigen.restriction is not None and v.size != self._v.size:
                _v = self.eigen.restriction.restrict_vector(_v)

        return _v
