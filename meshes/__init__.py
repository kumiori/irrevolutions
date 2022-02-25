# Copyright (C) 2020 Jørgen S. Dokken
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =========================================
# GMSH model to dolfinx.Mesh converter
# =========================================

from functools import wraps
import numpy
import gmsh

from mpi4py import MPI
from dolfinx.io import (
    extract_gmsh_geometry,
    extract_gmsh_topology_and_markers,
    ufl_mesh_from_gmsh,
)
from dolfinx.cpp.io import perm_gmsh, distribute_entity_data
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import create_meshtags, create_mesh

from gmsh import model


def mesh_bounding_box(mesh, i): return (
    min(mesh.geometry.x[:, i]), max(mesh.geometry.x[:, i]))

def read_from_msh(filename: str, cell_data=False, facet_data=False, gdim=None):
    """
    Reads a mesh from a msh-file and returns the dolfin-x mesh.
    Input:
        filename: Name of msh file
        cell_data: Boolean, True of a mesh tag for cell data should be returned
                   (Default: False)
        facet_data: Boolean, True if a mesh tag for facet data should be
                    returned (Default: False)
        gdim: Geometrical dimension of problem (Default: 3)
    """
    if MPI.COMM_WORLD.rank == 0:
        # Check if gmsh is already initialized
        try:
            current_model = gmsh.model.getCurrent()
        except ValueError:
            current_model = None
            gmsh.initialize()

        gmsh.model.add("Mesh from file")
        gmsh.merge(filename)
    output = gmsh_model_to_mesh(gmsh.model,
                                cell_data=cell_data,
                                facet_data=facet_data,
                                gdim=gdim)
    if MPI.COMM_WORLD.rank == 0:
        if current_model is None:
            gmsh.finalize()
        else:
            gmsh.model.setCurrent(current_model)
    return output

def gmsh_model_to_mesh(model, cell_data=False, facet_data=False, gdim=None, exportMesh=False, fileName="mesh.msh"):
    """
    Given a GMSH model, create a DOLFIN-X mesh and MeshTags. Can theoretically export to msh file
        model: The GMSH model
        cell_data: Boolean, True of a mesh tag for cell data should be returned
                   (Default: False)
        facet_data: Boolean, True if a mesh tag for facet data should be
                    returned (Default: False)
        gdim: Geometrical dimension of problem (Default: 3)
    """

    if gdim is None:
        gdim = 3

    if MPI.COMM_WORLD.rank == 0:
        # Get mesh geometry
        x = extract_gmsh_geometry(model)

        # Get mesh topology for each element
        topologies = extract_gmsh_topology_and_markers(model)

        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
        for i, element in enumerate(topologies.keys()):
            properties = model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {
                "id": element,
                "dim": dim,
                "num_nodes": num_nodes
            }
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = numpy.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)

        # Check for facet data and broadcast if found
        if facet_data:
            if tdim - 1 in cell_dimensions:
                num_facet_nodes = MPI.COMM_WORLD.bcast(
                    cell_information[perm_sort[-2]]["num_nodes"], root=0)
                gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
                marked_facets = numpy.asarray(
                    topologies[gmsh_facet_id]["topology"], dtype=numpy.int64)
                facet_values = numpy.asarray(
                    topologies[gmsh_facet_id]["cell_data"], dtype=numpy.int32)
            else:
                raise ValueError("No facet data found in file.")

        cells = numpy.asarray(topologies[cell_id]["topology"],
                              dtype=numpy.int64)
        cell_values = numpy.asarray(topologies[cell_id]["cell_data"],
                                    dtype=numpy.int32)

    else:
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = numpy.empty([0, num_nodes],
                               dtype=numpy.int32), numpy.empty([0, gdim])
        cell_values = numpy.empty((0, ), dtype=numpy.int32)
        if facet_data:
            num_facet_nodes = MPI.COMM_WORLD.bcast(None, root=0)
            marked_facets = numpy.empty((0, num_facet_nodes),
                                        dtype=numpy.int32)
            facet_values = numpy.empty((0, ), dtype=numpy.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)
    # Create MeshTags for cells
    if cell_data:
        local_entities, local_values = distribute_entity_data(
            mesh, mesh.topology.dim, cells, cell_values)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = AdjacencyList_int32(local_entities)
        ct = create_meshtags(mesh, mesh.topology.dim, adj,
                             numpy.int32(local_values))
        ct.name = "cells"

    # Create MeshTags for facets
    if facet_data:
        # Permute facets from MSH to Dolfin-X ordering
        # FIXME: This does not work for prism meshes
        facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())),
                                      mesh.topology.dim - 1, 0)
        gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
        marked_facets = marked_facets[:, gmsh_facet_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, mesh.topology.dim - 1, marked_facets, facet_values)
        mesh.topology.create_connectivity(mesh.topology.dim - 1,
                                          mesh.topology.dim)
        adj = AdjacencyList_int32(local_entities)
        ft = create_meshtags(mesh, mesh.topology.dim - 1, adj,
                             numpy.int32(local_values))
        ft.name = "facets"

    if exportMesh:
        gmsh.write(fileName)

    if cell_data and facet_data:
        return mesh, ct, ft
    elif cell_data and not facet_data:
        return mesh, ct
    elif not cell_data and facet_data:
        return mesh, ft
    else:
        return mesh


def get_tag(kwargs):
    return '' if (kwargs.get('tag') == None or kwargs.get('tag') == -1) else f"({kwargs.get('tag')})"


def geo_decorate_point(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        if kwargs.get('meshSize'):
            _str = f"Point {_tag} = {{ {args[0]}, {args[1]}, {args[2]}, {kwargs.get('meshSize')} }};"
        else:
            _str = f"Point {_tag} = {{ {args[0]}, {args[1]}, {args[2]}, {args[3]} }};"

        print(_str)
        return func(*args, **kwargs)
    return wrapper


def geo_decorate_line(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        print(
            f"Line {_tag} = {{ {args[0]}, {args[1]} }};")
        return func(*args, **kwargs)
    return wrapper


def geo_decorate_circle(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        print(
            f"Circle {_tag} = {{ {args[0]}, {args[1]}, {args[2]} }};")
        return func(*args, **kwargs)
    return wrapper


def geo_decorate_loop(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        print(
            f"Line Loop {_tag} = {{ {', '.join(map(str, args[0]))} }};")
        return func(*args, **kwargs)
    return wrapper


def geo_decorate_surface(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _str = [", ".join(map(str, arg)) for arg in args]
        _tag = get_tag(kwargs)
        print(
            f"Plane Surface {_tag} = {{ {', '.join(map(str, _str))} }};")
        return func(*args, **kwargs)
    return wrapper

def geo_decorate_physical(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        _str = " ".join(map(str, args[1]))
        if args[0] == 1:
            print(
                f"Physical Line {_tag} = {{ {_str} }};")
        elif args[0] == 2:
            print(
                f"Physical Surface {_tag} = {{ {_str} }};")

        return func(*args, **kwargs)
    return wrapper

_addPoint = geo_decorate_point(model.geo.addPoint)
_addLine = geo_decorate_line(model.geo.addLine)
_addCurveLoop = geo_decorate_loop(model.geo.addCurveLoop)
_addCircleArc = geo_decorate_circle(model.geo.addCircleArc)
_addPlaneSurface = geo_decorate_surface(model.geo.addPlaneSurface)
_addPhysicalGroup = geo_decorate_physical(model.geo.addPhysicalGroup)
