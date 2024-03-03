% Irreversible Solvers documentation master file, created by
% sphinx-quickstart on Sun Mar  3 17:21:07 2024.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# Irreversible Solvers • documentation

<!-- ```{include} ../../README.md
:relative-images:
``` -->

```{warning}
Theory under development
```
Three nonlinear variational solvers as modular components that address mathematical problems that are general enough to apply, in principle, to a large variety of applications that may arise in diverse fields of science including mechanics, technology, economy, social sciences, ecology, and quantum physics.


$$
\cup
$$

Our three nonlinear solvers address the solution of i) a variational inequality (akin to an obstacle problem), ii) a variational eigen-inequality in a vector space, and iii) a variational eigen-inequality in a (convex) cone, all the three under nonlinear constraints.


We implement a general framework to model and understand the competition  of smooth, continuous, incremental transitions and the observability of rare, brutal, discontinuous events.


What are the consequences of the well-posedness of a mathematical problem?


Picture "fracture", as a process: a paradigmatic occurrence of both modes of evolution whereby smooth, controlled, and incremental propagation of cracks are as commonplace as the sudden appearence of geometrically complex crack patterns, typically driven by brutal transitions.

Complex evolutionary systems challenge the axiom "Natura non facit saltus" [Leibniz]. 

## We solve:

$$
P(0):\text{ Given } T >0, \text{ find an } \text{ irreversible-constrained evolution } y_t$$
$$y_t: t\in [0, T]\mapsto X_t  
\text{ such that}$$ 
$$\text{[Unilateral Stability]} \qquad E(y_t) \leq E(y_t + z), \quad \forall z \in V_0 \times K^+_0\qquad [1]$$

Unpacked as:

$$
P_1(E): \text{Find }y_t\in X_t: E'(y_t)(z-y_t)\geq 0, \quad \forall z\in V_0\times K^+_0
$$
<!-- % P_2(E): \text{Find }\lambda, w \in \mathbb R\times X: E''(y_t)(w, z)=\lambda \langle w, z\rangle , \quad \forall z\in V\times K^+  -->

$$
P_2(E):\text{given } y_*, \text{find }(\mu_t, w) \in \mathbb R\times X_t: E''(y_*)(w, z-y_*)=\mu_t \langle w, z-y_*\rangle, \quad \forall z\in X_0 
$$


$$
P_2^+(E):\text{given } y_*, \text{find }(\lambda_t, {v}) \in \mathbb R/R^+\times X_t: E''(y_*)({v}, z-y_*) 
= \lambda_t  \langle {v}, z-y_*\rangle , \quad \forall z\in V_0\times K^+_0 
$$

$T$ defines a horizon of events for the system represented by its total energy $E$ and $X_t$ is the time-dependent space of admissible states. From a structural standpoint, a generic element of $X_t$ contains a macroscopic field that can be externally driven (or controlled, e.g. via boundary conditions) and an internal field (akin to an internal degree of order). As a consequence, $X_t$ has the structure of a cartesian product of two function spaces.


For further info, check out {doc}`usage`.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

usage
```

# Indices

- {ref}`genindex`
- {ref}`modindex`
