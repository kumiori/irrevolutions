from irrevolutions.meshes.boolean import create_disk_with_hole
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh
import numpy as np
from dolfinx import fem
from mpi4py import MPI
import dolfinx.io
import basix
from irrevolutions.solvers import SNESSolver
from dolfinx.fem import (
    Function,
    locate_dofs_topological,
    dirichletbc,
    assemble_scalar,
    form,
)
from irrevolutions.models import default_model_parameters
import os
from pathlib import Path
import yaml
import ufl
from petsc4py import PETSc
import sys
from irrevolutions.meshes.primitives import create_arc_ring_mesh

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    geom_params = {
        "R_outer": 1.0,  # Outer disk radius
        "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
        "lc": 0.05,  # Mesh element size
        "a": 0.1,  # Half-width of the refined region (-a < x < a)
        "geometric_dimension": 2,
    }
    # model, _ = create_arc_ring_mesh(0.5, 1.0, angle=180, lc=0.05)
    model, _ = create_arc_ring_mesh(geom_params)
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        model, comm, 0, gdim=geom_params["geometric_dimension"]
    )
    # gmsh_model, tdim = create_disk_with_hole(comm, geom_params)
    # mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)
    dx = ufl.Measure("dx", domain=mesh)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    with XDMFFile(MPI.COMM_WORLD, str(OUTPUT_DIR / "contact_hyper.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)

    with XDMFFile(
        mesh.comm, str(OUTPUT_DIR / "contact_hyper_facet_tags.xdmf"), "w"
    ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    element_u = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        degree=1,
        shape=((geom_params["geometric_dimension"]),),
    )

    V_u = dolfinx.fem.functionspace(mesh, element_u)

    u = dolfinx.fem.Function(V_u, name="Displacement")
    u_lb = dolfinx.fem.Function(V_u, name="Displacement_lb")
    u_ub = dolfinx.fem.Function(V_u, name="Displacement_ub")
    u_lb.interpolate(
        lambda x: np.stack([np.full_like(x[0], -np.inf), np.full_like(x[1], -np.inf)])
    )
    contact_tag = 60
    contact_dofs = dolfinx.fem.locate_dofs_topological(
        V_u, 1, facet_tags.find(contact_tag)
    )
    g_t = 1.1

    # UB on the contact boundary by meshtags
    # u_ub.interpolate(
    #     lambda x: np.stack([np.full_like(x[0], np.inf), np.full_like(x[1], np.inf)])
    # )
    # with u_ub.x.petsc_vec.localForm() as ub_local:
    #     ub_local[contact_dofs] = np.full_like(contact_dofs, g_t, dtype=PETSc.ScalarType)

    # UB on the entire field
    u_ub.interpolate(
        lambda x: np.stack([np.full_like(x[0], np.inf), np.full_like(x[1], g_t - x[1])])
    )

    # UB on the contact boundary by DOFs
    def on_outer_boundary(x):
        return np.isclose(x[0] ** 2 + x[1] ** 2, geom_params["R_outer"] ** 2, atol=1e-5)

    # Locate DOFs on the outer boundary
    outer_dofs = dolfinx.fem.locate_dofs_geometrical(V_u, on_outer_boundary)

    u_ub.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    print("Upper bound values:", u_ub.x.array)

    def bottom_boundary(x):
        return np.isclose(x[1], 0, atol=1e-2)

    bottom_dofs = locate_dofs_topological(
        V_u,
        mesh.topology.dim - 1,
        dolfinx.mesh.locate_entities_boundary(mesh, 1, bottom_boundary),
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(dir_path, "default_parameters.yml")
    with open(yaml_file, "r") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    # solver_parameters = parameters["solver"]
    solver_parameters = {
        "contact": {
            "type": "SNES",
            "prefix": "contact_",
            "snes": {
                # "snes_type": "vinewtonrsls",
                "snes_type": "vinewtonssls",
                "snes_linesearch_type": "basic",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "snes_atol": 1.0e-8,
                "snes_rtol": 1.0e-8,
                "snes_max_it": 50,
                "snes_monitor": "",
                "snes_vi_monitor": True,
            },
        }
    }
    E, nu = parameters["model"]["E"], parameters["model"]["nu"]

    body_f = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0], dtype=PETSc.ScalarType))

    # B = dolfinx.fem.Constant(
    #     mesh, np.array((0.0, 10), dtype=PETSc.ScalarType)
    # )  # Body force per unit volume
    B = dolfinx.fem.Function(V_u, name="Body_force")
    B.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.full_like(x[1], 1.0)]))
    T = dolfinx.fem.Constant(
        mesh, np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    )  # Traction force on the boundary
    # Kinematics
    d = geom_params["geometric_dimension"]
    I = ufl.Identity(d)  # Identity tensor  # noqa: E741
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    E = 10.0
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Invariants of deformation tensors
    Ic = ufl.tr(C)
    J = ufl.det(F)
    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    energy = psi * dx - ufl.dot(B, u) * dx - ufl.dot(T, u) * dx

    # F = ufl.derivative(energy, u)
    bottom_disp = dolfinx.fem.Constant(
        mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    )

    bcs_u = [
        dirichletbc(bottom_disp, bottom_dofs, V_u),
    ]

    for bc in bcs_u:
        bc.set(u_lb.x.array)
        bc.set(u_ub.x.array)

    # gather/scatter values
    for _u in [u_lb, u_ub]:
        _u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bcs = {"bcs_u": bcs_u}

    energy_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))

    opts = PETSc.Options()

    import petsc4py

    petsc4py.init(sys.argv)
    print(sys.argv)
    print(opts.getAll())

    contact = SNESSolver(
        energy_u,
        u,
        bcs.get("bcs_u"),
        petsc_options=solver_parameters.get("contact").get("snes"),
        bounds=(u_lb, u_ub),
        prefix=solver_parameters.get("contact").get("prefix"),
    )

    contact.solver.view()
    opts.view()

    loads = np.linspace(-1, 10.0, 30)

    for it, t in enumerate(loads):
        # u_ub.interpolate(
        #     lambda x: np.stack(
        #         [np.full_like(x[0], np.inf), np.full_like(x[1], gap - t)]
        #     )
        # )
        B.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.full_like(x[1], t)]))

        contact.solve()

        _B_work = assemble_scalar(dolfinx.fem.form(ufl.inner(B, u) * dx))

        print(f"Work done in this load step: {_B_work}")

        inactive_set = contact.solver.getVIInactiveSet()
        rnorm = contact.solver.getFunctionNorm()
        print(f"Final residual norm: {rnorm}")
        diff = np.linalg.norm(u.x.array[contact_dofs] - u_ub.x.array[contact_dofs])
        print(f"Max violation of contact constraint: {diff}")
        print("Max displacemnt on contact boundary:", np.max(u.x.array[contact_dofs]))
        print("Obstacle at:", np.unique(u_ub.x.array[contact_dofs]))
        # print("Lower bound values:", contact.lb.x.array)
        # print("Upper bound values:", contact.ub.x.array)
        print("SNES Solver Communicator:", contact.solver.getComm())
        print("MPI Rank:", MPI.COMM_WORLD.rank)

        with XDMFFile(MPI.COMM_WORLD, str(OUTPUT_DIR / "contact_hyper.xdmf"), "a") as xdmf:
            xdmf.write_function(u, t)
            xdmf.write_function(u_lb, t)
            xdmf.write_function(u_ub, t)
        print(f"Load step {it} done.")
