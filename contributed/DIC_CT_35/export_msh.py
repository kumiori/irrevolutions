#!/usr/bin/env python
# coding: utf-8


from dolfin import *
import os

filename = "mesh/DIC_running"


os.system("gmsh -2 " + filename + ".geo -format msh2")
os.system("dolfin-convert " + filename + ".msh " + filename + ".xml")
os.remove(filename + ".msh")
# xml to h5 (1-3)
mesh = Mesh(filename + ".xml")
#boundaries = MeshFunction("size_t", mesh, "mesh4_facet_region.xml")
subdomains = MeshFunction("size_t", mesh, filename + "_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, filename + "_facet_region.xml")


hdf = HDF5File(mesh.mpi_comm(), filename + ".h5", "w")
hdf.write(mesh, "/mesh")
hdf.write(boundaries, "/boundaries")
hdf.write(subdomains, "/subdomains")
hdf.close()


os.remove(filename + "_physical_region.xml")
os.remove(filename + "_facet_region.xml")
os.remove(filename + ".xml")


#mesh = Mesh()
#hdf = HDF5File(mesh.mpi_comm(), filename + ".h5", "r")
#hdf.read(mesh, "/mesh", False)
#ndim = mesh.topology().dim()

#boundaries = MeshFunction("size_t", mesh,1)
#hdf.read(boundaries, "/boundaries")

#subdomains = MeshFunction("size_t", mesh,2)
#hdf.read(subdomains, "/subdomains")
