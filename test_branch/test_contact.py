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
import yaml
import ufl
from petsc4py import PETSc


def create_arc_ring_mesh(inner_radius, outer_radius, angle=180, lc=0.1):
    """
    Create a 2D arc-shaped ring (annular sector) using Gmsh.

    Args:
        inner_radius (float): Inner radius of the arc.
        outer_radius (float): Outer radius of the arc.
        angle (float): Arc angle in degrees (e.g., 90 for a quarter ring).
        lc (float): Mesh characteristic length.

    Returns:
        Dolfinx mesh object.
    """
    gmsh.initialize()
    gmsh.model.add("ArcRing")

    # Convert angle to radians
    theta = np.radians(angle)

    # Create arc points
    _p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)
    p0 = gmsh.model.occ.addPoint(inner_radius, 0, 0, lc)
    p1 = gmsh.model.occ.addPoint(outer_radius, 0, 0, lc)
    p2 = gmsh.model.occ.addPoint(
        inner_radius * np.cos(theta), inner_radius * np.sin(theta), 0, lc
    )
    p3 = gmsh.model.occ.addPoint(
        outer_radius * np.cos(theta), outer_radius * np.sin(theta), 0, lc
    )
    tdim = 2  # 2D mesh
    # Create arcs
    arc_inner = gmsh.model.occ.addCircleArc(p0, _p0, p2)
    arc_outer = gmsh.model.occ.addCircleArc(p1, _p0, p3)
    line1 = gmsh.model.occ.addLine(p0, p1)
    line2 = gmsh.model.occ.addLine(p2, p3)

    # Create loop & surface
    loop = gmsh.model.occ.addCurveLoop([arc_inner, line2, -arc_outer, -line1])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    # Synchronize & generate mesh
    gmsh.model.occ.synchronize()
    model = gmsh.model
    surface_entities = [model[1] for model in model.getEntities(tdim)]

    domain = model.addPhysicalGroup(tdim, [surface], name="Domain")
    model.setPhysicalName(tdim, domain, "Surface")

    outer_boundary = gmsh.model.getBoundary([(2, surface)], oriented=False)
    fixed_edges = []
    for edge in outer_boundary:
        edge_id = edge[1]
        com = gmsh.model.occ.getCenterOfMass(1, edge_id)
        if np.isclose(com[1], 0.0, atol=1e-6):  # Select the bottom edge
            fixed_edges.append(edge_id)

    # **Define physical group for the bottom boundary**
    fixed_tag = gmsh.model.addPhysicalGroup(1, fixed_edges, name="Fixed_Boundary")
    gmsh.model.setPhysicalName(1, fixed_tag, "Fixed_Boundary")

    gmsh.model.mesh.generate(tdim)
    # Convert to Dolfinx mesh
    msh_file = "arc_ring.msh"
    gmsh.write(msh_file)
    # gmsh.finalize()
    tdim = 2  # 2D mesh

    # return dolfinx.io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, 0, gdim=2)
    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    geom_params = {
        "R_outer": 1.0,  # Outer disk radius
        "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
        "lc": 0.05,  # Mesh element size
        "a": 0.1,  # Half-width of the refined region (-a < x < a)
        "geometric_dimension": 2,
    }
    model, _ = create_arc_ring_mesh(0.5, 1.0, angle=180, lc=0.05)
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        model, comm, 0, gdim=geom_params["geometric_dimension"]
    )
    # gmsh_model, tdim = create_disk_with_hole(comm, geom_params)
    # mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)
    dx = ufl.Measure("dx", domain=mesh)
    with XDMFFile(MPI.COMM_WORLD, f"output/contact.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)

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
        # lambda x: np.full((x.shape[1], tdim), -np.inf)
        lambda x: np.stack([np.full_like(x[0], -np.inf), np.full_like(x[1], -x[1])])
    )
    g_t = 1.1

    u_ub.interpolate(
        lambda x: np.stack([np.full_like(x[0], np.inf), np.full_like(x[1], g_t - x[1])])
    )
    body_f = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0], dtype=PETSc.ScalarType))

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
            "prefix": "contact",
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
                "snes_vi_monitor": True,
                "snes_monitor": "",
            },
        }
    }
    E, nu = parameters["model"]["E"], parameters["model"]["nu"]

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return (
            E
            / (1 + nu)
            * (epsilon(v) + nu / (1 - nu) * ufl.tr(epsilon(v)) * ufl.Identity(2))
        )

    energy = 0.5 * ufl.inner(sigma(u), epsilon(u)) * dx - ufl.dot(body_f, u) * dx
    # F = ufl.derivative(energy, u)
    bottom_disp = dolfinx.fem.Constant(
        mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    )
    bcs_u = [
        # dirichletbc(top_disp, top_dofs, V_u),
        dirichletbc(bottom_disp, bottom_dofs, V_u),
    ]
    bcs = {"bcs_u": bcs_u}
    energy_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))
    contact = SNESSolver(
        energy_u,
        u,
        bcs.get("bcs_u"),
        petsc_options=solver_parameters.get("contact").get("snes"),
        bounds=(u_lb, u_ub),
        prefix=solver_parameters.get("contact").get("prefix"),
    )
    loads = np.linspace(-0.1, 1, 30)

    for it, t in enumerate(loads):
        # u_ub.interpolate(
        #     lambda x: np.stack(
        #         [np.full_like(x[0], np.inf), np.full_like(x[1], gap - t)]
        #     )
        # )
        body_f.value = np.array([0.0, -t], dtype=PETSc.ScalarType)
        contact.solve()
        # inactive_set = contact.solver.getVIInactiveSet()
        # inactive_set.setType("general")
        # __import__("pdb").set_trace()
        # inactive_set.setOptionsPrefix("inactive_set_")
        # inactive_set.setFromOptions()
        # inactive_set.getComm()
        # if inactive_set.getSize() > 0:
        #     inactive_set.view()
        # else:
        #     print("No inactive constraints detected!")

        # print(f"Number of inactive DOFs: {inactive_set.size}")
        # inactive_set.setComm(PETSc.COMM_WORLD)
        # inactive_set.view()

        rnorm = contact.solver.getFunctionNorm()
        print(f"Final residual norm: {rnorm}")
        diff = np.linalg.norm(u.x.array - u_ub.x.array)
        print(f"Max violation of contact constraint: {diff}")
        print("Lower bound values:", contact.lb.x.array)
        print("Upper bound values:", contact.ub.x.array)
        print("SNES Solver Communicator:", contact.solver.getComm())
        print("MPI Rank:", MPI.COMM_WORLD.rank)
        with XDMFFile(MPI.COMM_WORLD, f"output/contact.xdmf", "a") as xdmf:
            xdmf.write_function(u, t)
            # xdmf.write_function(u_lb, t)
            # xdmf.write_function(u_ub, t)
        print(f"Load step {it} done.")
