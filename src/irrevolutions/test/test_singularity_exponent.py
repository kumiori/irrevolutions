from sympy import *

x = symbols('x')

def singularity(omega):
    return nsolve(
        sin(2*x*(pi - omega)) + x*sin(2*(pi-omega)), 
        x, .5)
