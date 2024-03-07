---
title: 'Variational Solvers for Irreversile Evolutionary System'
tags:
  - Python
  - evolutions
  - stability
  - bifurcation
  - irreversibility
  - nonlinear
  - nonconvex
  - singular perturbations
authors:
  - name: Andrés A {León Baldelli}
    corresponding: true
    orcid: 0000-0002-3019-602X
    equal-contrib: false
    affiliation: 1
  - name: Pierluigi Cesana
    corresponding: true
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: $\partial$'Alembert Institute, CNRS, Sorbonne Universités, France
   index: 1
 - name: Kyushu University, Fukuoka, Japan
   index: 2
date: 4 March 2024
bibliography: paper.bib

---

# Summary

We study irreversible evolutionary processes with a general energetic notion of stability, and dedicate this contribution to releasing three nonlinear variational solvers as modular components that address mathematical problems that are general enough to apply, in principle, to a large variety of applications that may arise in diverse fields of science including mechanics, technology, economy, social sciences, ecology, and quantum physics.

Our three nonlinear solvers address the solution of i) a variational inequality (akin to an obstacle problem), ii) a variational eigen-inequality in a vector space, and iii) a variational eigen-inequality in a (convex) cone, all the three under nonlinear constraints. 

Our motivation proceeds from evolutionary problems in fracture mechanics, with the ultimate goal of deploying a transparent numerical platform for scientific validation and prediction of large scale natural fracture phenomena.


Our investigation rests upon the application of direct energy methods in the calculus of variations for the minimisation of energy/cost functionals which we reformulate for problems of the evolutionary type. In view of the applications, our solvers are used to show _one_ solution to a problem encoded in a system of two inequalities: one (pointwise almost-everywhere) constraint of irreversibility and one global energy statement. Ultimately, the latter plays the role of a selection principle.

As part of our commitment to open science, our solvers are released as FREE SOFTWARE, transparent and adaptable tools for a numerical platform devoted to the prediction of large-scale fracture events.

# Statement of need

Nonlinear systems provide rich models for processes ranging from the micro to macroscopic structural effects. In this context, the study of pattern formation provides insights into the fundamental processes that govern complex systems.

The emergence of patterns in evolutionary systems starting from homogeneous states is commonplace in biological morphogenesis (cellular organisation and cancer development), material science (phase transitions), physics of condensed matter (spin systems), in ecology and population dynamics (through species interactions), in economics, and networks. 


Quasi-static evolution problems arising in fracture and modeled via softening damage models are strongly nonlinear [@marigo:2023-la-mecanique], [@bourdin:2008-the-variational]. They can admit multiple solutions, or none [@leon-baldelli:2021-numerical] which demands both a functional theoretical framework and practical computational tools for real case scenarios.
Due to the lack of uniqueness for solutions, it is fundamental to leverage the full variational structure of the problem and investigate up to second order, to detect nucleation of stable modes and transitions of unstable states. The stability of a multiscale system along its nontrivial evolutionary paths in phase space is a key property that is difficult to check: numerically, at large scales with several material lengths involved, and analytically, in the infinite-dimensional setting.

Current available litterature of fracture mechanics limits investigation to unilateral first order variations, systematically neglecting the exploration of higher order information for critical points.

To fill this gap, our nonlinear solvers offer a flexible toolkit for advanced stability analysis of systems which evolve with constraints, based on a simple global-energetic unilateral criterion and functional mathematical constructs.

We include in this note a parametric benchmark test as a verification step to illustrate the accuracy, robustness, and reliability of our solvers, addressing a problem in one space dimension which well represents the main difficulties found in the applications of fracture. 


# Three solvers

`HybridSolver` (1) `BifurcationSolver,` (2) and ` StabilitySolver` (3) respectively implement the solution of three general purpose variational problems: 

1. A constrained variational inequality; 
    a first order necessary condition for incremental unilateral mechanical equilibrium.

2. A singular variational eigen-problem in a vector space;
    a bifurcation problem indicating uniqueness (or lack thereof) of the evolution path.
    
3. A constrained eigen-inequality in a convex cone;  
    originating from a second order eigenvalue problem indicating the stabilty of the state.

These solvers implement in a discrete setting the tools used to study general evolutionary problems naturally set in spaces of functions and formulated in terms of fully nonlinear functional operators in spaces of high or infinite dimension. In this context, systems can have surprising and complicated behaviours such as symmetry brealking bifurcations, endogenous pattern formation, localisations, and separation of scales.

The solvers discussed in this contribution have a general scope and can be extended or adapted to a variety of systems described by a unilateral stability law [1], by suitably replacing the energy expression, the definition constraints, and the description of the system's state.

We exploit the solvers to attack the following abstract problem which encodes a selection principle and a singularity: 
$$
P(0):\text{ Given } T >0, \text{ find an } \text{ irreversible-constrained evolution } y_t$$
$$y_t: t\in [0, T]\mapsto X_t  
\text{ such that}$$ 
$$\text{[Unilateral Stability]} \qquad E(y_t) \leq E(y_t + z), \quad \forall z \in V_0 \times K^+_0\qquad [1]$$


Above, $T$ defines a horizon of events for the system represented by its total energy $E$ and $X_t$ is the time-dependent space of admissible states. From a structural standpoint, a generic element of $X_t$ contains a macroscopic field that can be externally driven (or controlled, e.g. via boundary conditions) and an internal field (akin to an internal degree of order). As a consequence, $X_t$ has the structure of a cartesian product of two function spaces. In the applications of fracture, the kinematic variable is a vector-valued displacement $u(x)$ (for linearly-elastic models) and the degree of order $\alpha(x)$ represents the damage of the body/controls the softening of the material. Irreversibility applies to the internal variable, hence an <abbr >irreversible-constrained</abbr> evolution is an mapping parametrised by $t$ such that $\alpha(x)$ is non-decreasing with respect to $t$. Remark that the space of test functions is the mixed space given by the product of the homogeneous vector space $V_0$, associated to admissible variations of the kinematic field, times the space of perturbations of the internal order parameter $K^+_0$ which  only contains positive fields owing to the irreversibility constraint. The main difficulties in the problem above are correctly enforcing unilateral constraints and accounting for the changing nature of the space of variations. 


## Software

Our solvers are written in `Python` and are built on  `DOLFINx`, an expressive and performant parallel  distributed computing environment for solving partial differential equations using the finite element method [@dolfinx2023preprint] which enables us wrapping high-level functional mathematical constructs (e.g., energy functionals, boundary conditions, directional derivatives, ...) with full flexibility and configuration of the underlying linear algebra backend. We use PETSc [@petsc-user-ref], petsc4py [@dalcinpazklercosimo2011]  SLEPc.EPS [@hernandez:2005-slepc], and dolfiny [@Habera:aa] for parallel scalability.

Our solver API receives an abstract energy functional, a user-friendly description of the state of the system, and its associated constraints.

More precisely, both first and second order solvers are instantiated with the following arguments:
- **energy**: The total energy functional (a `ufl.form.Form`) associated with the system.
- **state**: A dictionary of `dolfinx.fem.function.Function` describing the current state variables.
- **bcs**: A dictionary of (lists of) boundary conditions for the unknown fields.
- **parameters**: A dictionary of numerical parameters for the solver (for default parameters, an empty dictionary).

The `HybridSolver` solves a (first order) constrained nonlinear variational inequality. The class implements a two-step hybrid strategy which is _ad hoc_ for energy models typical of applications in damage and fracture mechanics. The first phase (iterative alternate minimisation) is based on a de-facto _industry standard_ for damage models, conceived to exploit the (partial, directional) convexity of the underlying mechanical models. Once an approximate-solution enters the attraction set around a critical point (the attractor), the solver switches to perform a fully nonlinear step solving a block-matrix problem via Newton's method. This guarantees a precise estimation of the convergence of the first-order nonlinear problem based on the norm of the (constrained) residual.


The instantiation of `HybridSolver` involves the following additional parameters:
- **bounds**: A tuple specifying the bounds for the solution state variables. The default bounds are instances of `dolfinx.fem.function.Function`.
- **monitor**: An optional parameter for monitoring the solver's iterative progress (default: `None`).

- **nullspace** (optional): An optional parameter representing the nullspace of the linearised operator. Its default value is set to `None`.

More precisely, `BifurcationSolver` is a variational eigenvalue solver which uses SLEPc.EPS to explore the lower part of the spectrum of the nonlinear operator constructed as the Hessian of the energy, automatically computed performing two directional derivatives. Then, constraints are accounted for by projecting the full Hessian onto the subspace of inactive constraints [@jorge-nocedal:1999-numerical]. The relevance of this approach is typical of systems with threshold laws due to constraints. Thus, calling the method `def solve(alpha_old: dolfinx.fem.function.Function)`, `BifurcationSolver` returns a boolean value indicating whether the restricted Hessian is positive definite. Internally, the solver stores the lower part of the operators' spectrum as an array, the size of which can be specified in the parameters. For each mode, `bifurcation.spectrum[i]` is a dictionary containing the mode number $n$, the eigenvalue `lambda`, its numerical representation as a `petsc4py.PETSc.Vec` vector, and the approximate finite element fields `v, beta`.

Finally, `StabilitySolver` solves a constrained variational eigenvalue inequality in a convex cone, to check whether the (restricted) nonlinear Hessian operator is positive therein. This is done via the method
```python
def solve(
    alpha_old: dolfinx.fem.function.Function,
    z0: petsc4py.PETSc.Vec | None = None,
    inertia: tuple | None = None
) -> (bool)
```
which requires the order parameter field `alpha_old` at the previous timestep, an initial guess for the eigenvalue problem `z0`, and the `inertia` of the Hessian operator which is the tuple giving the number of (negative, zero, positive) eigenvalues. Starting from an initial guess $z_0^*$, it iteratively computes a series of (eigenvalue, eigenvector) pairs $\lambda_k, z_k$ converging to a limit $\lambda^*, z^*$ (as $k\to \infty$), by implementing a simple projection and scaling algorithm [@moreau:1962-decomposition, @pinto-da-costa:2010-cone-constrained].

The positivity of $\lambda^*$ (the smallest eigenvalue) allows to conclude on the stability  of the current state (or lack thereof), hence effectively solving P(0). Notice that, if the current state is unstable ($\lambda^*<1$), the minimal eigenmode indicates the direction of energy decrease. 

We dedicate a separate contribution to illustrate how the three solvers are algorithmically combined to solve problem P(0) in the case of fracture. \autoref{fig:convergence} illustrates the numerical convergence properties of the `StabilitySolver` algorithm, in  a representative one-dimensional verification test furhter detailed below.

![Rate of convergence for `StabilitySolver` (up to 1e-6 tolerance) in one spatial dimension (cf. benchmark problem below). The target quantities are the eigenvalue $\lim_k \lambda_k =: \lambda^*$ (pink) and the associated eigen-vector $x^*$ (error curve in blue). Note that, the residual dual vector (green) for the cone problem need not be zero at a solution, contrarily to the vector-space problem.\label{fig:convergence}](media/test_1d_stability-spa.pdf)


## Verification



To verify the numerical implementation we solve a nontrivial problem in one spatial dimension which is a proxy of the main difficulties of P(0) encountered in the applications.

We test a Rayleigh quotient $\mathcal R(z)$, namely we compute
$$
\min_{X_0} \mathcal R(z) \quad \text{and} \quad \min_{\mathcal K^+_0} \mathcal R(z) \qquad\qquad [2].$$
using `BifurcationSolver` and `StabilitySolver`, respectively.  The stability tests $[2]$ involve the computation of the minimum eigenvalues of  $\mathcal R(z)$. In connection with [1], here plays the role of a (rescaled) second variation of an energy. For definiteness, using the Sobolev spaces which are the natural setting for second order PDE problems, we set $X_0 = H^1_0(0, 1) \times H^1(0, 1)$ and $\mathcal K^+_0 = H^1_0(0, 1) \times \{\beta \in H^1(0, 1), \beta \geq 0\}$, defining
$$\mathcal R(z):= \dfrac{\int_0^1 a(\beta'(x))^2dx+\int_0^1 b(v'(x) -c\beta(x))^2dx}{\int_0^1\beta(x)^2dx},$$ 
 where $a, b, c$ are real coefficients such that $a>0, b>0, c\neq 0$. The expression above occurs, for instance, in the study of  the stability of a one-dimensional damageable bar, whereby $b$ is related to the spring constant of the material while $a, c$ encapsulate material, loading, and model parameters accounting for the coupling between elasticity and the order parameter, cf. the Appendix of [@pham:2011-the-issues] for completeness. We test our solvers against analytic solutions available in the 1d case, as key performance indicators we compare
i) The eigenspace and the profile of a typical nontrivial solution (cf. \autoref{fig:profile}),
ii) the measure of the support of solutions, parametrically, for a large choice of parameters $a, b,$ and $c$, (cf. \autoref{fig:phase_diag_D}).
iii) The optimal value of the quotient (i.e., the minimum), parametrically,  (cf. \autoref{fig:phase_diag_ball} and \autoref{fig:phase_diag_cone}).

![Comparison between profiles of solutions $\beta(x)$, minimisation in $X_0$ (left) vs. $\mathcal K^+_0$ (right). In the latter case, the solution $\beta(x)$ has support of size $D\in [0, 1]$.\label{fig:profile}](media/profile_comparison.pdf)


In the figures below, we report three tests of accuracy of the computations, comparing against analytic results the size of the support $D$ and the minimum value of $\mathcal R$, in the vector space and in the cone, parametrically for a large range of parameters. Numerical and analytical quantities are denoted with a $\#$ and with a $*$ sign, respectively.


![The size of the support of $\beta$ for the minimiser in the cone, the errorbars indicate absolute error with respect to closed form computation. We observe a clear separation between constant solutions with $D=1$ vs. nontrivial solutions. Where error bars are not shown, the error is none.\label{fig:phase_diag_D}](media/phase_diagram.pdf)


![Minimum value of the Rayleigh quotient, numerical computation vs. closed form result, of $R = \min_{X_0} \mathcal R$. Notice the separation in two regions separating constant solutions from nontrivial solutions. The outlier $(\pi^2a, bc^2)\sim (4, 0)$ represents a computation which did not reach convergence.\label{fig:phase_diag_ball}](media/phase_diagram_R_ball.pdf)

![Minimum value of the Rayleigh quotient, numerical computation vs. closed form result of $R = \min_{\mathcal K^+_0} \mathcal R$. The outlier $(\pi^2a, bc^2)\sim (4, 0)$ represents a computation which did not reach convergence. The mechanical interpretation is that only states with $R>1$ are energetically stable.\label{fig:phase_diag_cone}](media/phase_diagram_R_cone.pdf)


# Acknowledgements

We acknowledge contributions .........
the students of MEC647
Yves Capdeboscq, Jean-Jacques Marigo, Luc Nguyen, 

