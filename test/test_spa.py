import logging
import os
import pickle
import sys

import dolfinx
import irrevolutions.solvers.restriction as restriction
import numpy as np
import ufl
from dolfinx.io import XDMFFile
from irrevolutions.utils import _logger
from mpi4py import MPI
from test_cone_project import _cone_project_restricted

from . import test_binarydataio as bio

sys.path.append("../")


_logger.setLevel(logging.CRITICAL)


class NonConvergenceException(Exception):
    def __init__(self, message="Non-convergence error"):
        """
        Exception class for non-convergence errors during computations.
        """
        self.message = message
        super().__init__(self.message)


def load_minimal_constraints(filename, spaces):
    with open(filename, "rb") as file:
        minimal_constraints = pickle.load(file)

    # Assuming you have a constructor for your class

    reconstructed_obj = restriction.Restriction(spaces, np.array([[], []]))
    for key, value in minimal_constraints.items():
        setattr(reconstructed_obj, key, value)

    return reconstructed_obj


def test_spa():
    def iterate(x, xold, errors):
        """
        Perform convergence check and handle exceptions (NonConvergenceException).

        Args:
            x: Current vector.
            errors: List to store errors.

        Returns:
            bool: True if converged, False otherwise.
        """

        try:
            converged = _convergenceTest(x, xold, y)
        except NonConvergenceException as e:
            _logger.warning(e)
            _logger.warning("Continuing")
            # return False

        if not converged:
            # iteration += 1
            pass
        else:
            pass

        # should we iterate?
        return False if converged else True

    def update_lambda_and_y(xk, Ar):
        # Update λ_t and y computing:
        # λ_k = <x_k, A x_k> / <x_k, x_k>
        # y_k = A x_k - λ_k x_k
        _Axr = xk.copy()
        _y = xk.copy()
        Ar.mult(xk, _Axr)

        xAx_r = xk.dot(_Axr)

        _logger.debug("xk view in update at iteration")

        _lmbda_t = xAx_r / xk.dot(xk)
        _y.waxpy(-_lmbda_t, xk, _Axr)
        _y.norm()

        return _lmbda_t, _y

    def update_xk(xk, y, s):
        # Update _xk based on the scaling and projection algorithm
        xold = xk.copy()
        xk.copy(result=xold)
        # x_k = x_k + (-s * y)

        xk.axpy(-s, y)

        _cone_restricted = _cone_project_restricted(xk, _x, constraints)
        _cone_restricted.normalize()

        return _cone_restricted

    def _convergenceTest(x, xold, y=None):
        """
        Test the convergence of the current iterate xk against the prior, restricted version.

        Args:
            x: Current iterate vector.
            errors: List to store errors.

        Returns:
            bool: True if converged, False otherwise.
        """

        _atol = 1e-6
        _rtol = 1e-5
        _maxit = 1e5

        if iteration == _maxit:
            raise NonConvergenceException(
                f"SPA solver did not converge to atol {_atol} or rtol {_rtol} within maxit={_maxit} iteration."
            )

        diff = x.duplicate()
        diff.zeroEntries()
        # xdiff = x_old - x_k
        diff.waxpy(-1.0, xold, x)

        error_x_L2 = diff.norm()

        if y is not None:
            _residual_norm = y.norm()
        else:
            _residual_norm = 1

        errors.append(error_x_L2)

        if not iteration % 1000:
            _logger.critical(
                f"     [i={iteration}] error_x_L2 = {error_x_L2:.4e}, atol = {_atol}, res = {_residual_norm}"
            )

        _acrit = error_x_L2 < _atol
        if _acrit:
            _converged = True
        else:
            _converged = False

        return _converged

    comm = MPI.COMM_WORLD
    comm.Get_rank()
    comm.Get_size()

    _s = 1e-3

    with XDMFFile(
        comm, os.path.join(os.path.dirname(__file__), "data/input_data.xdmf"), "r"
    ) as file:
        mesh = file.read_mesh(name="mesh")

    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)

    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    ufl.Measure("dx", alpha.function_space.mesh)

    constraints = load_minimal_constraints("data/constraints.pkl", [V_u, V_alpha])

    bio.load_binary_matrix("data/A_hessian.mat")
    Ar = bio.load_binary_matrix("data/Ar_hessian.mat")
    x0 = bio.load_binary_vector("data/x0.vec")

    # zero vector, compatible with the linear system
    _x = x0.duplicate()

    # This throws a SIGSEGV
    # A.assemble()
    # Ar = constraints.restrict_matrix(A)
    x0r = constraints.restrict_vector(x0)

    xold = x0r.duplicate()  # x_k-1, =0 initially
    error = 1.0
    errors = []
    y = None
    iteration = 0
    data = {
        # "iterations": [],
        "error_x_L2": [],
        "lambda_k": [],
        # "lambda_0": [],
        "y_norm_L2": [],
        # "x_norm_L2": [],
    }

    while iterate(x0r, xold, error):
        iteration += 1
        x0r.copy(result=xold)

        lmbda_t, y = update_lambda_and_y(x0r, Ar)
        x0r = update_xk(x0r, y, _s)
        data["lambda_k"].append(lmbda_t)
        data["y_norm_L2"].append(y.norm())
        data["error_x_L2"].append(errors[-1])

    _logger.critical(
        f"lambda_0 = {lmbda_t:.4e}, residual norm = {y.norm(): .4e}, error = {errors[-1]: .4e}"
    )

    assert np.isclose(lmbda_t, -0.044659195907104675, atol=1e-4) == True


if __name__ == "__main__":
    test_spa()
