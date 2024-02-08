We suffered great loss today. Three weeks back: in time, we went one month back.
in Time.

What the fuck - simple as missing one, two, tree. 

LEFT BEHIND:
- work on scatters in tests
- implementation second order solver, parallel version
- parallel benchmarks
- postprocessing
- visualisation of fundamental modes

How many human/equivalemnt months worth? Does paying 
guarantee the outcome? What have we found? What have
we lost?

Count of the damages: 30K

This is a major pruning of the branches. 
The tree is alive as fuck. I show it in the app: The Code.

Agent 83252

rifare
-rw-r--r--  1 kumiori3  staff     457 Dec 15 13:01 test_Kproject.py
-rw-r--r--  1 kumiori3  staff   10195 Dec 15 13:01 test_cone_extracted.py
-rw-r--r--  1 kumiori3  staff    7307 Dec 15 13:02 test_cone_project.py
-rw-r--r--  1 kumiori3  staff    4783 Dec 15 13:02 test_extend.py
-rw-r--r--  1 kumiori3  staff    2681 Dec 15 13:02 test_extend_vector.py
-rw-r--r--  1 kumiori3  staff    7101 Dec 15 13:01 test_scatter.py
-rw-r--r--  1 kumiori3  staff   11502 Dec 15 13:02 test_scatter_MPI.py

rivedere
-rw-r--r--  1 kumiori3  staff   19204 Dec 15 13:01 test_linsearch.py
-rw-r--r--  1 kumiori3  staff    7226 Dec 15 13:01 test_stability.py

da pulire
-rw-r--r--  1 kumiori3  staff   24378 Dec 15 13:02 test_1d.py

-rw-r--r--  1 kumiori3  staff   38482 Dec 15 13:01 test_NLB.py
-rw-r--r--  1 kumiori3  staff    1625 Dec 15 13:01 test_V_notch_2D.py
-rw-r--r--  1 kumiori3  staff    6082 Dec 15 13:01 test_asymptotic_local_notch.py
-rw-r--r--  1 kumiori3  staff    7515 Dec 15 13:01 test_banquise1d.py
-rw-r--r--  1 kumiori3  staff   25876 Dec 15 13:01 test_computeFunctional.py
-rw-r--r--  1 kumiori3  staff   20690 Dec 15 13:01 test_discreteDamage-wild.py
-rw-r--r--  1 kumiori3  staff   25627 Dec 15 13:01 test_discreteDamage.py
-rw-r--r--  1 kumiori3  staff    5378 Dec 15 13:01 test_elasticity.py
-rw-r--r--  1 kumiori3  staff   20283 Dec 15 13:01 test_flow.py
-rw-r--r--  1 kumiori3  staff    1252 Dec 15 13:01 test_gmsh.py
-rw-r--r--  1 kumiori3  staff     923 Dec 15 13:01 test_gmsh2.py
-rw-r--r--  1 kumiori3  staff   11482 Dec 15 13:01 test_hybrid.py
-rw-r--r--  1 kumiori3  staff   11111 Dec 15 13:01 test_hybrid_serialised.py
-rw-r--r--  1 kumiori3  staff   16721 Dec 15 13:01 test_loadstep.py
-rw-r--r--  1 kumiori3  staff    1337 Dec 15 13:02 test_logging_mpi.py
-rw-r--r--  1 kumiori3  staff   11911 Dec 15 13:01 test_multifissa.py
-rw-r--r--  1 kumiori3  staff   10690 Dec 15 13:01 test_pacman.py
-rw-r--r--  1 kumiori3  staff    6315 Dec 15 13:01 test_plate_REG.py
-rw-r--r--  1 kumiori3  staff    4321 Dec 15 13:02 test_restriction.py
-rw-r--r--  1 kumiori3  staff    2140 Dec 15 13:02 test_sample_data.py
-rw-r--r--  1 kumiori3  staff     182 Dec 15 13:01 test_singularity_exponent.py
-rw-r--r--  1 kumiori3  staff     453 Dec 15 13:01 test_sympy.py
-rw-r--r--  1 kumiori3  staff    1679 Dec 15 13:01 test_tdcb_2D.py
-rw-r--r--  1 kumiori3  staff    7517 Dec 15 13:02 test_traction.py
-rw-r--r--  1 kumiori3  staff    4512 Dec 15 13:02 test_vi.py
-rw-r--r--  1 kumiori3  staff    2061 Dec 15 13:02 test_viz.py
➜  test git:(andres-conerecipe) ✗ 






DISCRETE: TRIGGER FIND FIRST OCCURRENCE
class SymbolicDiscreteDamage:

VIZ: 
For each load step I have a list of eigenvalues which I wish to represent in a scatter plot vertically aligned, at the same time step. How to do this?


LOGGING

User
is there a way to require a flush of logger messages to be able to sync information among MPI processes?






I am working with petsc vectors defined on a mixed functional space, and juggling with their block representation vs. the full vector representation.

The main feature of this approach is being able to swiftly select degrees of freedom based on pointwise constraints, in order to restrict the full (mixed vector-space) vector to a subset of our computational domain where constraints are active or inactive. 

The main object that allows to perform this reduction is called "constraints". As an object, it encapsulates the following informations:

        self.function_spaces = function_spaces
        self.blocal_dofs = blocal_dofs
        self.comm = self.function_spaces[0].mesh.comm

        self.bglobal_dofs_vec = []

        self.boffsets_vec = [0]
        offset_vec = 0

where blocal_dofs is a list of degrees of freedom, for each of the mixed function spaces, where constraints are active.

On the other hand, self.bglobal_dofs_vec is the list of degrees of freedom where constraints are active, the numbering referring to the full vector in the mixed space (thus the name global), per block. There also is the stacked version of the same list, namely self.bglobal_dofs_vec_stacked. 

Calling constraints.restrict_vector(v) returns a PETSc vector restricted to a proper subset of degrees of freedom as in:

    def restrict_vector(self, x: PETSc.Vec):
        arr = x.array[self.bglobal_dofs_vec_stacked]
        subx = PETSc.Vec().createWithArray(arr)

        return subx 
