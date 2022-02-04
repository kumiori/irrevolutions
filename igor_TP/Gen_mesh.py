#Numpy -> numerical library for Python. We'll use it for all array operations.
#It's written in C and it's faster (than traditional Python)
import numpy as np

#Yaml (Yet another markup language) -> We'll use it to pass, read and structure
#light text data in .yml files.
import yaml

#Json -> Another form to work with data. It comes from JavaScript. Similar functions
#that Yaml. Used speacily with API request, when we need data "fetch".
import json

#Communication with the machine:
#Sys -> allows to acess the system and launch commandes.
#Os - > allows to acess the operation system.
import sys
import os
from pathlib import Path

#Mpi4py -> Interface that allows parallel interoperability. MPI stands for' Message
#Passager Interface' and will be used to communicate computer nodes when lauching code
#in a parallel way

from mpi4py import MPI
#Petcs4py -> we use this library to handle with the data. Given acesses to solvers
import petsc4py
from petsc4py import PETSc

#Dolfinx
import dolfinx
import dolfinx.plot
from dolfinx import log
import logging

logging.basicConfig(level=logging.INFO)

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

#UFL (Unified Format Language) -> we'll be used to represent abstract way to 
#represent the language in a quadratic form
import ufl

#XDMFF -> format used for the output binary data
from dolfinx.io import XDMFFile

#Install 'gmsh' library -> we'll be used for the mesh.
#!{sys.executable}: to use the current kernel to make the installation 
import matplotlib.pyplot as plt

sys.path.append(cd '../')

import pdb

# meshes
import meshes
from meshes import gmsh_model_to_mesh, primitives

# visualisation
from utils import viz
import matplotlib.pyplot as plt
from utils.viz import plot_mesh, plot_vector, plot_scalar

#----------------------------------------
def mesh_V(
a,
h,
L,
gamma,
de,
de2,
key=0,
show=False,
filename='mesh.unv',
order = 1,
):
    """
    Create a 2D mesh of a notched three-point flexure specimen using GMSH.
    a = height of the notch
    h = height of the specimen
    L = width of the specimen
    gamma = notch angle
    de = density of elements at specimen
    de2 = density of elements at the notch and crack
    key = 0 -> create model for Fenicxs (default)
          1 -> create model for Cast3M
    show = False -> doesn't open Gmsh to vizualise the mesh (default)
           True -> open Gmsh to vizualise the mesh
    filename = name and format of the output file for key = 1 
    order = order of the function of form
    """
    gmsh.initialize()
    #gmsh.option.setNumber("General.Terminal",1)
    #gmsh.option.setNumber("Mesh.Algorithm",5)
    hopen = a*np.tan((gamma/2.0)*np.pi/180)
    c0 = h/10
    tdim = 2 
    
    model = gmsh.model()
    model.add('TPB')
    model.setCurrent('TPB')
    #Generating the points of the geometrie
    p0 = model.geo.addPoint(0.0, a, 0.0, de2, tag=0)
    p1 = model.geo.addPoint(hopen, 0.0, 0.0, de, tag=1)
    p2 = model.geo.addPoint(L/2, 0.0, 0.0, de, tag=2)
    p3 = model.geo.addPoint(L/2, h, 0.0, de, tag=3)
    p4 = model.geo.addPoint(0.0, h, 0.0, de, tag=4)
    if key == 0:
        p5 = model.geo.addPoint(-L/2, h, 0.0, de, tag=5)
        p6 = model.geo.addPoint(-L/2, 0.0, 0.0, de, tag=6)
        p7 = model.geo.addPoint(-hopen, 0.0, 0.0, de, tag=7)
    elif key == 1:
        p20 = model.geo.addPoint(0, a+c0, 0, de2, tag=20)
    #Creating the lines by connecting the points
    notch_right = model.geo.addLine(p0, p1, tag=8) 
    bot_right = model.geo.addLine(p1, p2, tag=9)
    right = model.geo.addLine(p2, p3, tag=10)
    top_right = model.geo.addLine(p3, p4, tag=11)
    if key == 0:
        top_left = model.geo.addLine(p4, p5, tag=12)
        left = model.geo.addLine(p5, p6, tag=13)
        bot_left = model.geo.addLine(p6, p7, tag=14)
        notch_left = model.geo.addLine(p7, p0, tag=15)
    elif key == 1:
        sym_plan = model.geo.addLine(p4, p20, tag=21)
        fissure = model.geo.addLine(p20, p0, tag=22)
    #Creating the surface using the lines created
    if key == 0:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, top_left, left, bot_left, notch_left])
    elif key == 1:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, sym_plan, fissure])
    surface = model.geo.addPlaneSurface([perimeter])
    #model.geo.addSurfaceLoop([surface,16])
    model.mesh.setOrder(order)
    
    #Creating Physical Groups to extract data from the geometrie
    if key == 0:
        gmsh.model.addPhysicalGroup(tdim-1, [left], tag = 101)
        gmsh.model.setPhysicalName(tdim-1, 101,'Left')

        gmsh.model.addPhysicalGroup(tdim-1, [right], tag=102)
        gmsh.model.setPhysicalName(tdim-1, 102,'Right')

        gmsh.model.addPhysicalGroup(tdim-2, [p6], tag=103)
        gmsh.model.setPhysicalName(tdim-2, 103,'Left_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=104)
        gmsh.model.setPhysicalName(tdim-2, 104,'Right_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=105)
        gmsh.model.setPhysicalName(tdim-2, 105, 'Load_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p0], tag=106)
        gmsh.model.setPhysicalName(tdim-2, 106, 'Notch_point')

        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')
    
    if key == 1:
        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')

        gmsh.model.addPhysicalGroup(tdim-1, [fissure], tag=111)
        gmsh.model.setPhysicalName(tdim-1, 111, 'fissure')

        gmsh.model.addPhysicalGroup(tdim-1, [sym_plan], tag=112)
        gmsh.model.setPhysicalName(tdim-1, 112, 'sym_plan')

        gmsh.model.addPhysicalGroup(tdim-2, [p20], tag=113)
        gmsh.model.setPhysicalName(tdim-2, 113, 'Crack_tip')

        gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=114)
        gmsh.model.setPhysicalName(tdim-2, 114, 'Load_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=115)
        gmsh.model.setPhysicalName(tdim-2, 115,'Right_point')   
    #Generating the mesh
    model.geo.synchronize()
    model.mesh.generate(tdim)
    if show:
        gmsh.fltk.run()
    #if key == 1:
    #    gmsh.write(filename)
    return gmsh.model

gmsh_Vnotch = mesh_V(0.00533, 0.0178, 0.0762, 90, 0.00533/5, 0.00533/10, key=1)

mesh = gmsh_model_to_mesh(gmsh_Vnotch, cell_data=True, facet_data=True, gdim=2, exportMesh=True, fileName='mesh.unv')

