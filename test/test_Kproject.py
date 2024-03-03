#!/usr/bin/env python3
import pdb
import numpy as np
from sympy import derive_by_array
import yaml
import json
from pathlib import Path
import sys
import os
import petsc4py
from petsc4py import PETSc
import dolfinx

COMM = PETSc.COMM_WORLD
N = 4

v = PETSc.Vec().createSeq(N)

v.array = np.random.rand(N) - 0.5

zero = v.duplicate()
zero.zeroEntries()
print("original", v.array)
v.pointwiseMax(v, zero)
print("pwise Max", v.array)
__import__("pdb").set_trace()
