import numpy as np
import sympy as sp


def solve_minimum(parameters):
    """
    Solve for the minimum of the fundamental quotient and return the corresponding substitutions.

    Parameters:
        parameters (dict): A dictionary containing the values for 'a', 'b', and 'c'.

    Returns:
        tuple: A tuple containing the minimum value and the substitutions for 'A' or 'C'.
    """
    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]
    A, C = sp.symbols("A C")
    _condition = b * c**2 < np.pi**2 * a
    print(f"bc**2 = {np.around(b*c**2, 2)}, π**2 * a = {np.around(np.pi**2 * a, 2)}")

    if _condition:
        print("case 1")
        _subs = {A: 0}
    elif not _condition:
        print("case 2")
        _subs = {C: 0}

    return np.min([b * c**2, np.pi**2 * a]), _subs


def solve_eigenspace_vector(parameters, idx=0):
    """
    Solve for the eigenspace in a vector space.

    Parameters:
        parameters (dict): A dictionary containing the values for 'a', 'b', and 'c'.
        idx (int): Index to choose the appropriate solution in case of multiple solutions.

    Returns:
        dict: A dictionary containing 'v', 'β', and 'D'.
    """
    x = sp.symbols("x", real=True)
    v = sp.Function("v", real=True)(x)
    β = sp.Function("β", real=True)(x)
    C, A = sp.symbols("C A")

    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]

    if b * c**2 < sp.pi**2 * a:
        print("case 1")
        _subs = {A: 0}
        A = 0
    elif b * c**2 > sp.pi**2 * a:
        print("case 2")
        _subs = {C: 0}
        C = 0

    β = C + A * sp.cos(sp.pi * x)
    v = c * A / sp.pi * sp.sin(sp.pi * x)

    depends_on_A = np.any(
        [sp.symbols("A") in expression.free_symbols for expression in [v, β]]
    )
    depends_on_C = np.any(
        [sp.symbols("C") in expression.free_symbols for expression in [v, β]]
    )

    _norm = sp.sqrt(
        np.sum([sp.integrate(eigenfunction**2, (x, 0, 1)) for eigenfunction in (v, β)])
    )

    print([expression.free_symbols for expression in [v, β]])
    print(_norm, depends_on_A, depends_on_C)

    if depends_on_A:
        print("depends_on_A")
        _normalise = [{sp.symbols("A"): ay} for ay in sp.solve(_norm - 1, A)]
    elif depends_on_C:
        print("depends_on_C")
        _normalise = [{sp.symbols("C"): cy} for cy in sp.solve(_norm - 1, C)]

    return {
        "v": v.subs(_normalise[idx]),
        "β": β.subs(_normalise[idx]),
        "D": 0,
    }, _normalise[idx]

    # return (v, β), _normalise


def solve_eigenspace_cone(parameters, idx=0):
    """
    Solve for the eigenspace in a convex set (cone).

    Parameters:
        parameters (dict): A dictionary containing the values for 'a', 'b', and 'c'.
        idx (int): Index to choose the appropriate solution in case of multiple solutions.

    Returns:
        dict: A dictionary containing 'v', 'β', and 'D'.
    """
    x = sp.symbols("x", real=True)
    sp.Function("v", real=True)(x)
    β = sp.Function("β", real=True)(x)
    C, A = sp.symbols("C A")

    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]
    D = 0

    if b * c**2 < sp.pi**2 * a:
        print("case 1")
        β = C

    elif b * c**2 > sp.pi**2 * a:
        print("case 2")
        # D = sp.symbols('D')
        D = (sp.pi**2 * a / (b * c**2)) ** (1 / 3)
        β = sp.Piecewise(
            (C * (1 + sp.cos(sp.pi * x / D)), (0 <= x) & (x <= D)), (0, True)
        )

        _min = (np.pi**2 * a) ** (1 / 3) * (b * c**2) ** (2 / 3)

    elif b * c**2 == sp.pi**2 * a:
        print("case eq")
        _min = b * c**2
        _subs = {C: 0}
        C = 0
        β = C + A * sp.cos(sp.pi * x)
        # abs(A) < C

    depends_on_A = sp.symbols("A") in β.free_symbols
    depends_on_C = sp.symbols("C") in β.free_symbols
    depends_on_D = sp.symbols("D") in β.free_symbols

    _norm = sp.sqrt(sp.integrate(β**2, (x, 0, 1)))

    # print([expression.free_symbols for expression in [v, β]])
    print(_norm)

    if depends_on_A:
        print("depends_on_A")
        _normalise = [{sp.symbols("A"): ay} for ay in sp.solve(_norm - 1, A)]
    elif depends_on_C:
        print("depends_on_C")
        _normalise = [{sp.symbols("C"): cy} for cy in sp.solve(_norm - 1, C) if cy > 0]
    elif depends_on_D:
        print("depends_on_D")

    return {"v": 0, "β": β.subs(_normalise[idx]), "D": D}, _normalise[idx]


def book_of_the_numbers(scale_b=3, scale_c=3):
    """This function, informally called `fuck_dgsi`, invokes the book of the numbers
    to get three real quantities, according to the scriptures.

    @article{pham:2011-the-issues,
        author = {Pham, Kim and Marigo, Jean-Jacques and Maurini, Corrado},
        date-added = {2015-08-24 14:23:19 +0000},
        date-modified = {2022-08-10 11:03:49 +0200},
        journal = {Journal of the Mechanics and Physics of Solids},
        number = {6},
        pages = {1163--1190},
        publisher = {Elsevier},
        title = {The issues of the uniqueness and the stability of the homogeneous response in uniaxial tests with gradient damage models},
        volume = {59},
        year = {2011},
        }

    Also, fuck Elsevier and Springer Nature.

    """
    while True:
        a = np.random.rand()
        b = np.random.rand() * scale_b
        c = float(
            (np.random.choice([-1, 1], 1) * np.random.rand(1))[0] * scale_c
        )  # Generate a random number with sign between 0 and 3

        # Check conditions
        if a > 0 and b > 0 and c != 0:
            break

    return {"a": a, "b": b, "c": c}
