#!/usr/bin/env python3
import numpy as np
import ufl
from petsc4py import PETSc


def rotated_stiffness_matrix(theta_deg=-30.0, C11=1.0, C12=0.5, C44=198.25):
    theta = theta_deg * np.pi / 180.0

    c11 = PETSc.ScalarType(C11)
    c12 = PETSc.ScalarType(C12)
    c44 = PETSc.ScalarType(C44)

    cmat = np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c44]])

    transform = np.array(
        [
            [
                np.cos(theta) ** 2,
                np.sin(theta) ** 2,
                2.0 * np.cos(theta) * np.sin(theta),
            ],
            [
                np.sin(theta) ** 2,
                np.cos(theta) ** 2,
                -2.0 * np.cos(theta) * np.sin(theta),
            ],
            [
                -np.cos(theta) * np.sin(theta),
                np.cos(theta) * np.sin(theta),
                np.cos(theta) ** 2 - np.sin(theta) ** 2,
            ],
        ]
    )

    return transform @ cmat @ transform.T


def anisotropic_dissipated_energy(alpha, Gc, ell, theta_deg=-30.0):
    cmatr = rotated_stiffness_matrix(theta_deg=theta_deg)

    cr11 = cmatr[0, 0]
    cr12 = cmatr[0, 1]
    cr14 = cmatr[0, 2]
    cr44 = cmatr[2, 2]

    kappa_tensor = ufl.sym(ufl.grad(alpha))
    kappa = ufl.as_vector([kappa_tensor[0, 0], kappa_tensor[1, 1], kappa_tensor[0, 1]])

    cr_voigt = ufl.as_matrix(
        [
            [cr11, cr12, 2.0 * cr14],
            [cr12, cr11, -2.0 * cr14],
            [2.0 * cr14, -2.0 * cr14, 4.0 * cr44],
        ]
    )

    return (5.0 / 96.0) * Gc * (
        ufl.inner(kappa, cr_voigt * kappa) * ell**3 + alpha / ell
    ) * ufl.dx


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    print(rotated_stiffness_matrix())
