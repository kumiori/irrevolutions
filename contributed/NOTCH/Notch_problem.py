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
sys.path.append('../') #-> this serves to add a path to the code search for things
import os
from pathlib import Path

#pdb -> usefull for debugging, it can stop a code operation and allows to read 
#variables and do calculations
import pdb
#pdb.set_trace() #-> point of stop for debugging

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
import gmsh

import matplotlib.pyplot as plt

# meshes
import meshes
from meshes import primitives

# visualisation
from irrevolutions.utils import viz
import matplotlib.pyplot as plt
from utils.viz import plot_mesh, plot_vector, plot_scalar
import pyvista
from pyvista.utilities import xvfb

# Parameters

parameters = {
    #In case of evolution (nonlinear) problems, it's necessary to define a max
    #and a min. For the elastic solution, just one value in needed.
    'loading': {
        'type':'ID', #ID -> Imposed Displacement | IF -> Imposed Force
        'min': 0,
        'max': 1.5,
        'steps': 20
    },
    'geometry': {
        'geom_type': 'bar',
        'Lx': 1.,
        'Ly': 0.01
    },
    'model': {
        'E': 1.0,
        'nu': 0.3,
        'mu': 0, #don't change it -> calculated later
        'lmbda': 0, #don't change it -> calculated later
        'w1': 1.,
        'ell': 0.01,
        'k_res': 1.e-8
    },
    'solvers': {
          'elasticity': {        
            'snes': {
                'snes_type': 'newtontr',
                'snes_stol': 1e-8,
                'snes_atol': 1e-8,
                'snes_rtol': 1e-8,
                'snes_max_it': 100,
                'snes_monitor': "",
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            }
        },
          'damage': {        
            'snes': {
                'snes_type': 'vinewtonrsls',
                'snes_stol': 1e-5,
                'snes_atol': 1e-5,
                'snes_rtol': 1e-8,
                'snes_max_it': 100,
                'snes_monitor': "",
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            },
        },
                'damage_elasticity': {
                "max_it": 100,
                "alpha_rtol": 1.0e-5,
                "criterion": "alpha_H1"
            }
    }
}
E = parameters["model"]["E"]
poisson = parameters["model"]["nu"]
parameters['model']['lmbda'] =E*poisson/((1+poisson)*(1-2*poisson))
parameters['model']['mu'] = E/(2*(1+poisson))
# parameters.get('loading') -> this parameters can be defined and obtained from
# a external file. In the first exemple (mec647_VI_1), the parameters were
# read from a .yml file.


def mesh_V(
a,
h,
L,
n,
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
    n = width of the load interface
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
    gmsh.option.setNumber("General.Terminal",1)
    gmsh.option.setNumber("Mesh.Algorithm",5)
    hopen = a*np.tan((gamma/2.0)*np.pi/180)
    c0 = h/40
    load_len = n
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
        #Load facet
        p21 = model.geo.addPoint(load_len, h, 0.0, de, tag=30)
        p22 = model.geo.addPoint(-load_len, h, 0.0, de, tag=31)
    elif key == 1:
        p20 = model.geo.addPoint(0, a+c0, 0, de2, tag=20)
    #Creating the lines by connecting the points
    notch_right = model.geo.addLine(p0, p1, tag=8) 
    bot_right = model.geo.addLine(p1, p2, tag=9)
    right = model.geo.addLine(p2, p3, tag=10)
    #top_right = model.geo.addLine(p3, p4, tag=11)
    if key == 0:
        top_right = model.geo.addLine(p3, p21, tag=11)
        top_left = model.geo.addLine(p22, p5, tag=12)
        left = model.geo.addLine(p5, p6, tag=13)
        bot_left = model.geo.addLine(p6, p7, tag=14)
        notch_left = model.geo.addLine(p7, p0, tag=15)
        #Load facet
        load_right = model.geo.addLine(p21, p4, tag=32)
        load_left = model.geo.addLine(p4, p22, tag=33)
    elif key == 1:
        top_right = model.geo.addLine(p3, p4, tag=11)
        sym_plan = model.geo.addLine(p4, p20, tag=21)
        fissure = model.geo.addLine(p20, p0, tag=22)
    #Creating the surface using the lines created
    if key == 0:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, load_right, load_left, top_left, left, bot_left, notch_left])
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

        gmsh.model.addPhysicalGroup(tdim-1, [load_right], tag=107)
        gmsh.model.setPhysicalName(tdim-1, 107, 'load_right')

        gmsh.model.addPhysicalGroup(tdim-1, [load_left], tag=108)
        gmsh.model.setPhysicalName(tdim-1, 108, 'load_left')

        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')
   
    #Cast3M can't read Physical Groups of points (dim = 0). Instead, we check the number in the mesh and input in manually in the code.
    #The number of a node doesn't change if it's in a point of the geometry
    if key == 1:
        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')

        gmsh.model.addPhysicalGroup(tdim-1, [fissure], tag=111)
        gmsh.model.setPhysicalName(tdim-1, 111, 'fissure')

        gmsh.model.addPhysicalGroup(tdim-1, [sym_plan], tag=112)
        gmsh.model.setPhysicalName(tdim-1, 112, 'sym_plan')

        #gmsh.model.addPhysicalGroup(tdim-2, [p20], tag=113)
        #gmsh.model.setPhysicalName(tdim-2, 113, 'Crack_tip')

        #gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=114)
        #gmsh.model.setPhysicalName(tdim-2, 114, 'Load_point')

        #gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=115)
        #gmsh.model.setPhysicalName(tdim-2, 115,'Right_point')   
    #Generating the mesh
    model.geo.synchronize()
    model.mesh.generate(tdim)
    if show:
        gmsh.fltk.run()
    if key == 1:
        gmsh.write(filename)
    return gmsh.model

a=0.075
h=0.3
n=1/50
L=1
gamma = 90
de = a/20
de2 = a/40
gmsh_model = mesh_V(a, h, L, n, gamma, de, de2)
#In this moment, it could be necessary to get the data of the cells and facets
#In this case, we are taking info of the facets so that we can define subdomains
#in order to apply Newman Bondary conditions, which means that is a condition 
#applied not in the displacement (variable of interest), but in the correspond
#variable (in this case, force/pressure)
mesh,facet_tags = meshes.gmsh_model_to_mesh(gmsh_model,
                                          cell_data=False,
                                          facet_data=True,
                                          gdim=2)


#Plot mesh
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")

# Functional setting
#'u' represents the displacement in this problem. In order to solve it, the 
#continuos field  'u' is replaced by a discrite form u = som[vec(function_forme)
#*vec(nodal_displacement)]
#In order to define the vec(function_forme), the ufl library is used. 

#A VectorElement represents a combination of basic elements such that each
#component of a vector is represented by the basic element. The size is usually
#omitted, the default size equals the geometry dimension.

#ulf.VectorElement(<Type of the element>, <Geometry of the element>, 
#degree=<Degree of element: 1 - Linear, 2 - Quadratic, etc.>, dim= <Target 
#dimension of the element: 1 - Line, 2 - Area, 3 - Volume>)

#Lagrange is a familly type of elements -> polynomial functions of forme;
#The Lagrange elements are going to be defined in the mesh as such we take the
#geometry of elements present in the mesh.

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(),
                              degree=1, dim=2)
element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                              degree=1)

#After defining the Finite Element in ufl, a association with dolfinx is made.
#To inputs are necessary, the mesh and the element type created. In some sense, 
#we obtain the "discretised model w/ elements definied".
V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

#In this model, we also defines functions necessaries to solve the problem. 
#This functions are definied in the entire space/model.
u = dolfinx.fem.Function(V_u, name="Displacement") #The discrete nodal valeus of
                                                   #the displacement
u_ = dolfinx.fem.Function(V_u, name="BC_Displacement")
u_imposed = dolfinx.fem.Function(V_u, name="Imposed_Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
# Bounds -> the values of alpha must be max([0,1],[alpha(t-1),1]) 
alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

#In order to defined a function in a specific subspace of the model, it must be 
#specified in the model 'V_u.sub(i)', where i = 0 -> x, 1 -> y, 2-> z.
#Don't forget to collapse, to choose only the DOF associated with the subspace.

#I don't think  this part works to definy the body force applied in a geometry.
#It could be better to define it in the energy definition as constant. If not a 
#constant, we might need to define as a space function.
# g = dolfinx.fem.Function(V_u, name="Body_pressure")
# with g.vector.localForm() as loc:
#   loc.set(-78500.0)

# Integral measures -> in order to define the energy lately, it's necessary to 
#define the integral measures, as such one is a integral.
dx = ufl.Measure("dx", domain=mesh) #-> volume measure
#We include here the subdomain data generated at the gmsh file.
ds = ufl.Measure("ds", subdomain_data = facet_tags, domain=mesh) #-> surface measure
#ds(<number of the facet tags>)
#dS = ufl.Measure("dS", domain = mesh) - inner boundaries of the mesh -> not usefull

import models
from models import DamageElasticityModel as Brittle

model = Brittle(parameters.get('model'))
state = {'u': u, 'alpha': alpha}
#The total energy density is calculated this time using a already written 
#function of the "model". This return the elasticity energy (with the a(alpha))
#and the damage energy term. To count for externals forces, it need to substract it
#from the total energy
total_energy = model.total_energy_density(state) * dx #- ufl.dot(force,u)*ds(107) - ufl.dot(force,u)*ds(108)
if parameters['loading']['type'] == 'ID':
  total_energy = model.total_energy_density(state) * dx
if parameters['loading']['type'] == 'IF':
  #Getting load parameters
  force = dolfinx.fem.Function(V_u, name="Contact_force")
  loading_force = -1*parameters['loading']['max']
  force.interpolate(lambda x: (np.zeros_like(x[0]), loading_force*np.ones_like(x[1])))
  total_energy = model.total_energy_density(state) * dx - ufl.dot(force,u)*ds(107) - ufl.dot(force,u)*ds(108)

# Boundary sets
#Function that returns 'TRUE' if the point of the mesh is in the region you want
#to apply the BC.
def BC_points(x):
  #x[0] is the vector of X-coordinate of all points ; x[1] is the vector of Y-coordinate
  return np.logical_and(
      np.logical_or(np.isclose(x[0],-L/2),np.isclose(x[0],L/2)),
      np.isclose(x[1],0))
BC_entities = dolfinx.mesh.locate_entities_boundary(mesh, 0, BC_points)
BC_dofs = dolfinx.fem.locate_dofs_topological(V_u, 0, BC_entities)
u_.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))

#FOR IMPOSED FORCE :
if parameters['loading']['type'] == 'IF':
  bcs_u = [dirichletbc(u_, BC_dofs)]
#FOR IMPOSED DISPLACEMENT :
if parameters['loading']['type'] == 'ID':
  def ID_points(x):
    return np.logical_and(np.equal(x[1],h), 
                          np.logical_and(np.greater_equal(x[0],-1*n),
                                                          np.less_equal(x[0],n)
                                                          ))
  ID_entities = dolfinx.mesh.locate_entities_boundary(mesh, 0, ID_points)
  ID_dofs = dolfinx.fem.locate_dofs_topological(V_u, 0, ID_entities)
  u_imposed.interpolate(lambda x: (np.zeros_like(x[0]), -1*parameters['loading']['max']*np.ones_like(x[1])))
  bcs_u = [dirichletbc(u_, BC_dofs),dirichletbc(u_imposed,ID_dofs)]

dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], -L/2))
dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], L/2))
BC_dofs_alpha = dolfinx.fem.locate_dofs_topological(V_alpha, 0, BC_entities)
if parameters['loading']['type'] == 'IF':
  bcs_alpha = [
              dirichletbc(np.array(0., dtype = PETSc.ScalarType),
                          BC_dofs_alpha,
                          V_alpha)
  ]
if parameters['loading']['type'] == 'ID':
  ID_dofs_alpha = dolfinx.fem.locate_dofs_topological(V_alpha, 0, ID_entities)
  bcs_alpha = [
              dirichletbc(np.array(0., dtype = PETSc.ScalarType),
                          np.concatenate([dofs_alpha_left, dofs_alpha_right, BC_dofs_alpha, ID_dofs_alpha]),
                          V_alpha)
  ]
bcs_alpha=[]
#dofs_alpha_left, dofs_alpha_right
bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

# Update the bounds
set_bc(alpha_ub.vector, bcs_alpha)
set_bc(alpha_lb.vector, bcs_alpha)

import algorithms
from algorithms import am
solve_it = am.AlternateMinimisation(total_energy, 
                         state, 
                         bcs, 
                         parameters.get("solvers"), 
                         bounds=(alpha_lb,alpha_ub))

#solve_it.elasticity
#Loop for evolution
Loads = np.linspace(parameters.get("loading").get("min"),
                    parameters.get("loading").get("max"),
                    parameters.get("loading").get("steps"))

data = {
    'elastic': [],
    'surface': [],
    'total': [],
    'load': []
}

for (i_t,t) in enumerate(Loads):
  #update bondary conditions
  if parameters['loading']['type'] == 'ID':
    u_imposed.interpolate(lambda x: (np.zeros_like(x[0]), -1*t*np.ones_like(x[1])))
  if parameters['loading']['type'] == 'IF':
    force.interpolate(lambda x: (np.zeros_like(x[0]), loading_force*t*np.ones_like(x[1])))
  #update lower bound for damage
  alpha.vector.copy(alpha_lb.vector)
  #solve for current load step
  solve_it.solve()
  #postprocessing
  #global
  surface_energy = assemble_scalar(dolfinx.fem.form(model.damage_energy_density(state)*dx))
  elastic_energy = assemble_scalar(dolfinx.fem.form(model.elastic_energy_density(state)*dx))
  
  data.get('elastic').append(elastic_energy)
  data.get('surface').append(surface_energy)
  data.get('total').append(surface_energy+elastic_energy)
  data.get('load').append(t)
  
  print(f'Solved timestep {i_t}, load {t}')
  print(f'Elastic energy {elastic_energy:.3g}, Surface energy {surface_energy:.3g}')

  #saving


plt.plot(data.get('load'), data.get('surface'), label='surface')
plt.plot(data.get('load'), data.get('elastic'), label='elastic')
#plt.plot(data.get('load'), [1./2. * t**2*L for t in data.get('load')], label='anal elast', ls=':', c='k')

plt.title('My specimen')
plt.legend()
#plt.yticks([0, 1/20], [0, '$1/2.\sigma_c^2/E_0$'])
#plt.xticks([0, 1], [0, 1])

try:
    from dolfinx.plot import create_vtk_mesh as compute_topology
except ImportError:
    from dolfinx.plot import create_vtk_topology as compute_topology
    
def plot_scalar(alpha, plotter, subplot=None, lineproperties={}):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = alpha.function_space
    mesh = V.mesh
    
    # topology, cell_types = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    # topology, cell_types = dolfinx.plot.create_vtk_topology(
        # mesh, mesh.topology.dim)
    topology, cell_types = compute_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)

    plotter.subplot(0, 0)
    grid.point_data["alpha"] = alpha.compute_point_values().real
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    return plotter

# postprocessing
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )
_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt.screenshot(f"alpha.png")

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 1),
    )
# plt = plot_scalar(u.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 0))

_plt.screenshot(f"displacement_MPI.png")


plt.figure()
plt.plot(data.get('load'), data.get('surface'), label='surface')
plt.plot(data.get('load'), data.get('elastic'), label='elastic')
#plt.plot(data.get('load'), [1./2. * t**2*L for t in data.get('load')], label='anal elast', ls=':', c='k')

plt.title('My specimen')
plt.legend()
plt.savefig('energy.png')
