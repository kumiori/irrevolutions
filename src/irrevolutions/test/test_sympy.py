from sympy import cos, diff, lambdify, symbols

Θ = symbols("Θ")
func = Θ**3.0

fp = lambdify(Θ, diff(func, Θ), "numpy")
fpp = lambdify(Θ, diff(func, Θ, 2), "numpy")
fppp = lambdify(Θ, diff(func, Θ, 3), "numpy")

for i, f in enumerate([fp, fpp, fppp]):
    print(f"f^({i})(1.) =", f(1.0))

λ = 1.0

g = cos((1 + λ) * Θ)

for i in range(3):
    _f = lambdify(Θ, diff(g, Θ, i), "numpy")
    print(f"g^({i})(1.) =", _f(1.0))
