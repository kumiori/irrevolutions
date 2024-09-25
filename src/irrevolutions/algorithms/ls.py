import logging
import random

import mpi4py
import numpy as np
from dolfinx.cpp.log import LogLevel, log
from dolfinx.fem import Function, assemble_scalar, form
from petsc4py import PETSc
from irrevolutions.utils import norm_H1

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class LineSearch:
    """A class to perform line search algorithms for variational problems.

    Attributes:
        energy (ufl.Form): The total energy functional.
        state (dict): The current state of the system with displacement ('u') and damage ('alpha').
        parameters (dict): Line search parameters such as method and interpolation order.
    """

    def __init__(self, energy, state, linesearch_parameters={}):
        """
        Initialize LineSearch with energy functional, state, and optional parameters.

        Parameters:
            energy (ufl.Form): The total energy functional.
            state (dict): The current state containing the displacement ('u') and damage ('alpha').
            linesearch_parameters (dict): Parameters for the line search algorithm.
        """
        super(LineSearch, self).__init__()

        self.energy = energy
        self.state = state
        self.parameters = linesearch_parameters

        self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space
        self.mesh = self.V_u.mesh

        self.u0 = Function(state["u"].function_space)
        self.alpha0 = Function(state["alpha"].function_space)

    def search(self, state, perturbation, interval, m=2, method="min"):
        """
        Perform a line search using a polynomial interpolation method.

        Parameters:
            state (dict): The current state with displacement ('u') and damage ('alpha').
            perturbation (dict): Perturbation fields containing the displacement perturbation ('v') and damage perturbation ('beta').
            interval (tuple): A tuple specifying the admissible search interval (hmin, hmax).
            m (int): The interpolation order (default is 2).
            method (str): The method for line search ('min' for minimum, 'random' for random choice).

        Returns:
            tuple: The optimal step size h_opt, energies along the search direction, the fitted polynomial, and coefficients.
        """
        v = perturbation["v"]
        beta = perturbation["beta"]

        u_0 = Function(state["u"].function_space)
        alpha_0 = Function(state["alpha"].function_space)

        state["u"].x.petsc_vec.copy(u_0.x.petsc_vec)
        state["alpha"].x.petsc_vec.copy(alpha_0.x.petsc_vec)

        en_0 = assemble_scalar(form(self.energy))

        hmin, hmax = interval
        htest = np.linspace(hmin, hmax, np.int32(m + 1))
        energies_1d = []
        perturbation_norms = []

        for h in htest:
            with state["u"].x.petsc_vec.localForm() as u_local, state["alpha"].x.petsc_vec.localForm() as alpha_local:
                u_local.array[:] = u_0.x.petsc_vec.array[:] + h * v.x.petsc_vec.array[:]
                alpha_local.array[:] = alpha_0.x.petsc_vec.array[:] + h * beta.x.petsc_vec.array[:]

            state["u"].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            state["alpha"].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            yh_norm = np.sum([norm_H1(func) for func in state.values()])
            perturbation_norms.append(yh_norm)

            en_h = assemble_scalar(form(self.energy))
            energies_1d.append(en_h - en_0)

        # Restore original state
        u_0.x.petsc_vec.copy(state["u"].x.petsc_vec)
        alpha_0.x.petsc_vec.copy(state["alpha"].x.petsc_vec)

        # Polynomial fit and optimal step computation
        z = np.polyfit(htest, energies_1d, m)
        p = np.poly1d(z)

        if m == 2:
            logging.info("Line search using quadratic interpolation")
            h_opt = -z[1] / (2 * z[0])
        else:
            logging.info(f"Line search using polynomial interpolation (order {m})")
            h = np.linspace(0, hmax, 30)
            h_opt = h[np.argmin(p(h))]

        if method == "random":
            h_opt = random.uniform(hmin, hmax)

        return h_opt, energies_1d, p, z

    def perturb(self, state, perturbation, h):
        """
        Apply a perturbation to the state fields (displacement and damage).

        Parameters:
            state (dict): The current state with displacement ('u') and damage ('alpha').
            perturbation (dict): Perturbation fields for displacement ('v') and damage ('beta').
            h (float): The step size for the perturbation.

        Returns:
            dict: Updated state after applying the perturbation.
        """
        v = perturbation["v"]
        beta = perturbation["beta"]

        z0_norm = np.sum([norm_H1(func) for func in state.values()])

        with state["u"].x.petsc_vec.localForm() as u_local, state["alpha"].x.petsc_vec.localForm() as alpha_local:
            u_local.array[:] = u_local.array[:] + h * v.x.petsc_vec.array[:]
            alpha_local.array[:] = alpha_local.array[:] + h * beta.x.petsc_vec.array[:]

        state["u"].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        state["alpha"].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        zh_norm = np.sum([norm_H1(func) for func in state.values()])

        logging.critical(f"Initial state norm: {z0_norm}")
        logging.critical(f"Perturbation norm: {zh_norm}")

        return state

    def admissible_interval(self, state, perturbation, alpha_lb, bifurcation):
        """
        Compute the admissible interval for line search based on the current state and bifurcation condition.

        Parameters:
            state (dict): The current state.
            perturbation (dict): Perturbation fields for displacement ('v') and damage ('beta').
            alpha_lb (Function): Lower bound of damage.
            bifurcation (tuple): Bifurcation data.

        Returns:
            tuple: The admissible interval (hmin, hmax) for the line search.
        """
        alpha = state["alpha"]
        beta = bifurcation[1]

        one = max(1.0, max(alpha.x.petsc_vec[:]))
        mask = np.int32(np.where(beta.x.petsc_vec[:] > 0)[0])

        hp2 = (one - alpha.x.petsc_vec[mask]) / beta.x.petsc_vec[mask] if len(mask) > 0 else [np.inf]
        hp1 = (alpha_lb.x.petsc_vec[mask] - alpha.x.petsc_vec[mask]) / beta.x.petsc_vec[mask] if len(mask) > 0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        mask_neg = np.int32(np.where(beta.x.petsc_vec[:] < 0)[0])
        hn2 = (one - alpha.x.petsc_vec[mask_neg]) / beta.x.petsc_vec[mask_neg] if len(mask_neg) > 0 else [-np.inf]
        hn1 = (alpha_lb.x.petsc_vec[mask_neg] - alpha.x.petsc_vec[mask_neg]) / beta.x.petsc_vec[mask_neg] if len(mask_neg) > 0 else [np.inf]
        hn = (max(hn2), min(hn1))

        hmax = np.array(np.min([hp[1], hn[1]]))
        hmin = np.array(np.max([hp[0], hn[0]]))

        hmax_glob = np.array(0.0, "d")
        hmin_glob = np.array(0.0, "d")

        comm.Allreduce(hmax, hmax_glob, op=mpi4py.MPI.MIN)
        comm.Allreduce(hmin, hmin_glob, op=mpi4py.MPI.MAX)

        hmax = float(hmax_glob)
        hmin = float(hmin_glob)

        if hmin > 0 or hmax == 0 and hmin == 0 or hmax < hmin:
            log(LogLevel.INFO, "Line search failed.")
            return 0.0, 0.0

        assert hmax > hmin, "hmax must be greater than hmin"

        return hmin, hmax
    
    def get_unilateral_interval(self, state, perturbation):
        """
        Compute the admissible interval for line search based on the cone-problem. 
        This is a unilateral interval, where the upper bound is determined by the condition:
            h: alpha + h*perturbation <= 1

        Parameters:
            state (dict): The current state with displacement ('u') and damage ('alpha').
            perturbation (dict): Perturbation fields for displacement ('v') and damage ('beta').

        Returns:
            tuple: A tuple containing (0, hmax), where hmax is the maximum allowable step size.
        """
        alpha = state["alpha"]
        beta = perturbation["beta"]

        # Ensure that the perturbation is non-negative
        assert (beta.x.petsc_vec[:] >= 0).all(), "beta must be non-negative"

        # Compute the upper bound for the admissible interval
        one = max(1.0, max(alpha.x.petsc_vec[:]))
        mask = np.int32(np.where(beta.x.petsc_vec[:] > 0)[0])

        _hmax = (
            (one - alpha.x.petsc_vec[mask]) / beta.x.petsc_vec[mask]
            if len(mask) > 0
            else [np.inf]
        )

        # Reduce the global maximum value for parallel processing
        hmax_glob = np.array(0.0, "d")
        comm.Allreduce(np.min(_hmax), hmax_glob, op=mpi4py.MPI.MIN)

        hmax = float(hmax_glob)
        assert hmax > 0, "hmax must be greater than 0"

        return (0, hmax)


class StabilityStepper:
    """Iterator for handling stability steps in a quasistatic simulation of an evolution process.
    
    This is an quasistatic implementation of a time stepper based on a variational stability
    statement. Its key feature is the ability to pause the time stepper during stability transitions.

    """

    def __init__(self, loads):
        """
        Initialize the StabilityStepper.

        Parameters:
            loads (list or array-like): The list of load steps for the simulation.
        """
        self.i = 0
        self.stop_time = False
        self.loads = loads

    def __iter__(self):
        """Return self as the iterator."""
        return self

    def __next__(self):
        """Move to the next load step, pausing if required."""
        logger.info(f"\n\nCalled next, can time be stopped? {self.stop_time}")

        if self.stop_time:
            # If time is paused, return the current index without incrementing
            self.stop_time = False
            index = self.i
        else:
            # If not paused, check if there are more items to return
            if self.i < len(self.loads):
                self.i += 1
                index = self.i
            else:
                raise StopIteration

        return index

    def pause_time(self):
        """Pause the time stepper for manual intervention or other reasons."""
        self.stop_time = True
        logger.info(f"Called pause, stop_time is {self.stop_time}")