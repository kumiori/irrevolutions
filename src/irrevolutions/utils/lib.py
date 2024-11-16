import numpy as np
import sympy as sp


def singularity_exp(omega):
    """Exponent of singularity, λ\in [1/2, 1]
    lmbda : = sin(2*lmbda*(pi - omega)) + lmbda*sin(2(pi-lmbda)) = 0"""
    from sympy import nsolve, pi, sin, symbols

    x = symbols("x")

    return nsolve(sin(2 * x * (pi - omega)) + x * sin(2 * (pi - omega)), x, 0.5)


#  = parameters["material"]
def _local_notch_asymptotic(x, ω=45, t=1.0, par={}):
    from sympy import cos, pi, sin, symbols

    λ = singularity_exp(ω)
    Θ = symbols("Θ")
    _E = par["E"]
    ν = par["ν"]
    Θv = np.arctan2(x[1], x[0])

    coeff = ((1 + λ) * sin((1 + λ) * (pi - ω))) / ((1 - λ) * sin((1 - λ) * (pi - ω)))

    _f = (
        (2 * np.pi) ** (λ - 1)
        * (cos((1 + λ) * Θ) - coeff * cos((1 - λ) * Θ))
        / (1 - coeff)
    )

    f = sp.lambdify(Θ, _f, "numpy")
    fp = sp.lambdify(Θ, sp.diff(_f, Θ, 1), "numpy")
    fpp = sp.lambdify(Θ, sp.diff(_f, Θ, 2), "numpy")
    fppp = sp.lambdify(Θ, sp.diff(_f, Θ, 3), "numpy")

    r = np.sqrt(x[0] ** 2.0 + x[1] ** 2.0)
    _c1 = (λ + 1) * (1 - ν * λ - ν**2.0 * (λ + 1))
    _c2 = 1 - ν**2.0
    _c3 = 2.0 * (1 + ν) * λ**2.0 + _c1
    _c4 = _c2
    _c5 = λ**2.0 * (1 - λ**2.0)

    ur = t * (r**λ / _E * (_c1 * f(Θv) + _c2 * fpp(Θv))) / _c5
    uΘ = t * (r**λ / _E * (_c3 * fp(Θv) + _c4 * fppp(Θv))) / _c5
    _tdim = 2
    values = np.zeros((_tdim, x.shape[1]))
    values[0] = ur * np.cos(Θv) - uΘ * np.sin(Θv)
    values[1] = ur * np.sin(Θv) + uΘ * np.cos(Θv)
    return values
