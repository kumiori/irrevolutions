# Copyright (C) 2020 Jørgen S. Dokken
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =========================================
# GMSH model to dolfinx.Mesh converter
# =========================================

from functools import wraps

# from dolfinx.mesh import create_meshtags, create_mesh
from gmsh import model


def mesh_bounding_box(mesh, i):
    return (min(mesh.geometry.x[:, i]), max(mesh.geometry.x[:, i]))


def get_tag(kwargs):
    return (
        ""
        if (kwargs.get("tag") is None or kwargs.get("tag") == -1)
        else f"({kwargs.get('tag')})"
    )


def geo_decorate_point(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        if kwargs.get("meshSize"):
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
        print(f"Line {_tag} = {{ {args[0]}, {args[1]} }};")
        return func(*args, **kwargs)

    return wrapper


def geo_decorate_circle(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        print(f"Circle {_tag} = {{ {args[0]}, {args[1]}, {args[2]} }};")
        return func(*args, **kwargs)

    return wrapper


def geo_decorate_loop(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        print(f"Line Loop {_tag} = {{ {', '.join(map(str, args[0]))} }};")
        return func(*args, **kwargs)

    return wrapper


def geo_decorate_surface(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _str = [", ".join(map(str, arg)) for arg in args]
        _tag = get_tag(kwargs)
        print(f"Plane Surface {_tag} = {{ {', '.join(map(str, _str))} }};")
        return func(*args, **kwargs)

    return wrapper


def geo_decorate_physical(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _tag = get_tag(kwargs)
        _str = " ".join(map(str, args[1]))
        if args[0] == 1:
            print(f"Physical Line {_tag} = {{ {_str} }};")
        elif args[0] == 2:
            print(f"Physical Surface {_tag} = {{ {_str} }};")

        return func(*args, **kwargs)

    return wrapper


_addPoint = geo_decorate_point(model.geo.addPoint)
_addLine = geo_decorate_line(model.geo.addLine)
_addCurveLoop = geo_decorate_loop(model.geo.addCurveLoop)
_addCircleArc = geo_decorate_circle(model.geo.addCircleArc)
_addPlaneSurface = geo_decorate_surface(model.geo.addPlaneSurface)
_addPhysicalSurface = geo_decorate_physical(model.geo.addPhysicalGroup)
