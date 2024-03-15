#!/usr/bin/env python3
import numpy as np
from petsc4py import PETSc

COMM = PETSc.COMM_WORLD
N = 4

v = PETSc.Vec().createSeq(N)

v.array = np.random.rand(N)-.5

zero = v.duplicate()
zero.zeroEntries()
print('original', v.array)
v.pointwiseMax(v, zero)
print('pwise Max', v.array)