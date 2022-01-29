"""
Function to create input for large loop to compute minimum energy for different ell values
Works by exporting into a txt file --> Could plot directly, avoided in order to not need unnecessary recomputations when changing plotting settings

Input
Data from parameter files
Output
Textfile with parameterresults
"""

import numpy as np

import dolfinx
import dolfinx.plot
import dolfinx.io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml
from dolfinx.fem.assemble import assemble_scalar

sys.path.append("../")
from solvers import SNESSolver

petsc4py.init(sys.argv)

import logging

logging.basicConfig(level=logging.INFO)

with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


Lx = parameters.get("geometry").get("Lx")
Ly = parameters.get("geometry").get("Ly")
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]],
                                     [100, 10],
                                     cell_type=CellType.triangle)
V = FunctionSpace(mesh, ("CG", 1))

zero = Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = Function(V)
with one.vector.localForm() as loc:
    loc.set(1.0)


def left(x):
    is_close = np.isclose(x[0], 0.0)
    return is_close


def right(x):
    is_close = np.isclose(x[0], Lx)
    return is_close


left_facets = dolfinx.mesh.locate_entities_boundary(mesh,
                                                    mesh.topology.dim - 1,
                                                    left)
left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1,
                                                left_facets)

right_facets = dolfinx.mesh.locate_entities_boundary(mesh,
                                                     mesh.topology.dim - 1,
                                                     right)
right_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1,
                                                 right_facets)

bcs = [dirichletbc(zero, left_dofs), dirichletbc(one, right_dofs)] #I think we might not have any Dirichlet BC on RHS
u = Function(V)
if parameters.get("model").get("ell")==None:
    ell_list=np.linspace(parameters.get("model").get("ellMin"), parameters.get("model").get("ellMax"), num=parameters.get("model").get("ellIter"))
else:
    ell_list=[parameters.get("model").get("ell")]
resArry=np.zeros((2, len(ell_list)))
i=0
for ell in ell_list:
    energy = (ell * ufl.inner(ufl.grad(u), ufl.grad(u)) + u / ell) * ufl.dx
    denergy = ufl.derivative(energy, u, ufl.TestFunction(V))
    ddenergy = ufl.derivative(denergy, u, ufl.TrialFunction(V))

    problem = SNESSolver(
        denergy,
        u,
        bcs,
        bounds=(zero, one),
        petsc_options=parameters.get("solvers").get("damage").get("snes"),
        prefix="vi",
    )

    solver_snes = problem.solver
    solver_snes.setType("vinewtonrsls")

    solver_snes.setTolerances(rtol=1.0e-8, max_it=250)
    solver_snes.getKSP().setType("preonly")
    solver_snes.getKSP().setTolerances(rtol=1.0e-8)
    solver_snes.getKSP().getPC().setType("lu")


    def monitor(snes, its, fgnorm):
        if(its%10==0):
            print(f"Iteration {its:d}, error: {fgnorm:2.3e}")


    solver_snes.setMonitor(monitor)
    solver_snes.solve(None, u.vector)
    
    min_en = assemble_scalar(dolfinx.fem.form(energy))

    resArry[0, i]=ell
    resArry[1, i]=min_en
    print(min_en)
    i+=1
np.savetxt("energyEll.txt", resArry)
