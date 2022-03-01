import gmsh
import numpy as np
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
