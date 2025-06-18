from irrevolutions.algorithms.gf import JumpSolver

from dolfinx import fem, nls, la
from petsc4py import PETSc
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import (
    functionspace,
    Function,
    dirichletbc,
    locate_dofs_topological,
    Constant,
)
import basix

import sys
import petsc4py
import pyvista
from pathlib import Path

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD
from irrevolutions.utils.viz import plot_scalar, plot_vector

import os

from copy import deepcopy

from dolfinx.fem.petsc import LinearProblem

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = os.path.join(outdir, "test_jump")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    x = ufl.SpatialCoordinate(mesh)
    h = 1e-2  # projection step size

    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    V_u = functionspace(mesh, element_u)
    V_alpha = functionspace(mesh, element_alpha)

    u = Function(V_u, name="u")
    alpha = Function(V_alpha, name="alpha")
    alpha_0 = Function(V_alpha, name="alpha0")
    alpha_h = Function(V_alpha, name="alpha_h")
    grad_proj_fn = Function(V_alpha, name="projected_gradient")
    f_expr = ufl.sin(2 * 3 * ufl.pi * x[0]) * ufl.sin(2 * 3 * ufl.pi * x[1])

    alpha.x.array[:] = 0.1  # initial value
    alpha.x.petsc_vec.copy(result=alpha_0.x.petsc_vec)

    beta = Function(V_alpha, name="beta")
    # sin_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    sin_expr = ufl.sin(2 * ufl.pi * x[1])
    beta_expr = fem.Expression(sin_expr, V_alpha.element.interpolation_points())
    beta.interpolate(beta_expr)
    beta.x.petsc_vec.array[:] = np.maximum(beta.x.petsc_vec.array, 0.0)
    # normalise beta
    beta.x.petsc_vec.array[:] /= np.linalg.norm(beta.x.petsc_vec.array, ord=2)

    # Simple energy functional: just a penalty on alpha
    def energy_function(u, alpha):
        # return -ufl.inner(alpha, alpha) * ufl.dx
        return f_expr * alpha * ufl.dx

    # perturb alpha by beta and assemble the energy gradient
    alpha.x.petsc_vec.copy(result=alpha_h.x.petsc_vec)
    alpha_h.x.petsc_vec.axpy(h, beta.x.petsc_vec)
    alpha_h.x.scatter_forward()

    dE_alpha = fem.petsc.assemble_vector(
        fem.form(
            ufl.derivative(
                energy_function(u, alpha_h), alpha_h, ufl.TestFunction(V_alpha)
            )
        )
    )
    grad_proj = dE_alpha.copy()
    grad_proj.array[:] = np.minimum(-grad_proj.array, 0.0)
    grad_proj.copy(result=grad_proj_fn.x.petsc_vec)
    grad_proj_fn.x.scatter_forward()
    # Projected gradient
    state = {"u": u, "alpha": alpha}
    perturbation = {"v": u, "beta": beta}

    bcs = []  # no boundary conditions
    jump_parameters = {"tau": 1e-1, "max_steps": 100, "rtol": 1e-6, "verbose": True}

    flow = JumpSolver(energy_function, state, bcs, jump_parameters)
    state = flow.solve(perturbation, h=0.01)

    print(
        "Final alpha min/max:",
        state["alpha"].x.array.min(),
        state["alpha"].x.array.max(),
    )

    if comm.Get_size() == 1:
        # xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        plotter = pyvista.Plotter(
            title="alpha",
            window_size=[1600, 600],
            shape=(2, 2),
        )
        plotter, grid_alpha = plot_scalar(
            alpha, plotter, scalars_name="alpha", subplot=(0, 0)
        )
        plotter, _ = plot_scalar(beta, plotter, scalars_name="beta", subplot=(0, 1))
        plotter, _ = plot_scalar(
            grad_proj_fn,
            plotter,
            scalars_name="projected_gradient",
            subplot=(1, 0),
        )

        plotter.subplot(1, 1)
        grid_grad = deepcopy(grid_alpha)
        grid_grad.point_data["alpha_elevated"] = grid_grad.point_data.pop("alpha")
        grid_grad.set_active_scalars("alpha_elevated")

        elevated = grid_grad.warp_by_scalar("alpha_elevated")
        plotter.add_mesh(
            elevated, scalars="alpha_elevated", show_edges=True, cmap="coolwarm"
        )
        zero_plane = pyvista.Plane(
            center=(0.5, 0.5, 0.0), direction=(0, 0, 1), i_size=1.0, j_size=1.0
        )
        plotter.add_mesh(zero_plane, color="gray", style="wireframe", opacity=0.6)

        plotter.screenshot(f"{prefix}/alpha.png")
        __import__("pdb").set_trace()
