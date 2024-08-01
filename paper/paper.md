---
title: 'Variational Solvers for Irreversible Evolutionary Systems'
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
    orcid: 0000-0002-0304-9382
    corresponding: true
    orcid: 0000-0002-0304-9382
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: $\partial$'Alembert Institute, CNRS, Sorbonne Universités, Place Jussieu 75252 Paris Cedex 05, France
   index: 1
 - name: Institute of Mathematics for Industry, Kyushu University, 744 Motooka, Nishi-ku, Fukuoka, 819-0395, Japan.
   index: 2
date: 8 March 2024
bibliography: paper.bib

---

# Summary

We study irreversible evolutionary processes with a general energetic notion of stability. We dedicate this contribution to releasing three nonlinear variational solvers as modular components (based on FEniCSx/dolfinx) that address three mathematical optimisation problems. They are general enough to apply, in principle, to evolutionary systems with instabilities, jumps, and emergence of patterns which is commonplace in diverse arenas spanning from quantum to continuum mechanics, economy, social sciences, and ecology. Our motivation proceeds from fracture mechanics, with the ultimate goal of deploying a transparent numerical platform for scientific validation and prediction of large scale natural fracture phenomena. Our solvers are used to compute _one_ solution to a problem encoded in a system of two inequalities: one (pointwise almost-everywhere) constraint of irreversibility and one global energy statement. **~~As part of our commitment to open science, our solvers are released as free software.~~**

# Statement of need

Quasi-static evolution problems arising in fracture are strongly nonlinear [@marigo:2023-la-mecanique], [@bourdin:2008-the-variational]. They can admit multiple solutions, or none [@leon-baldelli:2021-numerical]. This demands both a functional theoretical framework and practical computational tools for real case scenarios. Due to the lack of uniqueness of solutions, it is fundamental to leverage the full variational structure of the problem and investigate up to second order, to detect nucleation of stable modes and transitions of unstable states. The stability of a multiscale system along its nontrivial evolutionary paths in phase space is a key property that is difficult to check: numerically, for real case scenarios with several length scales involved, and analytically, in the infinite-dimensional setting. The current literature in computational fracture mechanics predominantly focuses on unilateral first-order criteria, systematically neglecting the exploration of higher-order information for critical points.

To fill this gap, our nonlinear solvers offer a flexible toolkit for advanced stability analysis of systems which evolve with constraints.


# **Functionality**

`HybridSolver` (1) `BifurcationSolver,` (2) and `StabilitySolver` (3) implement the solution of three general purpose variational problems: 

1. A constrained variational inequality; that is first order necessary conditions for unilateral equilibrium.

2. A singular variational eigen-problem in a vector space; that is
    a bifurcation problem indicating uniqueness (or lack thereof) of the evolution path.
    
3. A constrained eigen-inequality in a convex cone; originating from a second order eigenvalue problem indicating stabilty of the system (or lack thereof).

These numerical tools can be used to study general evolutionary problems formulated in terms of fully nonlinear functional operators in spaces of high or infinite dimension. In this context, systems can have surprising and complicated behaviours such as symmetry breaking bifurcations, endogenous pattern formation, localisations, and separation of scales. Our solvers can be extended or adapted to a variety of systems described by an energetic principle (or unilateral stability law, see [1] below).

We exploit the solvers to attack the following abstract problem which encodes a selection principle: 
$$
P(0):\text{ Given } T >0, \text{ find an } \text{ irreversible-constrained evolution } y_t$$
$$y_t: t\in [0, T]\mapsto X_t  
\text{ such that}$$ 
$$\text{[Unilateral Stability]} \qquad E(y_t) \leq E(y_t + z), \quad \forall z \in V_0 \times K^+_0\qquad [1]$$


Above, $T$ defines a horizon of events. The system is represented by its total energy $E$ and $X_t$ is the time-dependent space of admissible states. A generic element of $X_t$ contains a macroscopic field that can be externally driven (or controlled, e.g. via boundary conditions) and an internal field (akin to an internal degree of order). In the applications of fracture, the kinematic variable is a vector-valued displacement $u(x)$ and the degree of order $\alpha(x)$ controls the softening of the material. Irreversibility applies to the internal variable, hence an <abbr >irreversible-constrained</abbr> evolution is a mapping parametrised by $t$ such that $\alpha_t(x)$ is non-decreasing with respect to $t$. **The kinematic variable is subject to bilateral variations belonging to a linear subset of a Sobolev vector space $V_0$, whereas ** ~~Remark that~~ the test space for the internal order parameter $K^+_0$ only contains positive fields owing to the irreversibility constraint. The main difficulties are to correctly enforce unilateral constraints and to account for the changing nature of the space of variations. 


## Software

Our solvers are written in `Python` and are built on  `DOLFINx`, an expressive and performant parallel  distributed computing environment for solving partial differential equations using the finite element method [@dolfinx2023preprint]. It enables us wrapping high-level functional mathematical constructs with full flexibility and control of the underlying linear algebra backend. We use PETSc [@petsc-user-ref], petsc4py [@dalcinpazklercosimo2011], SLEPc.EPS [@hernandez:2005-slepc], and dolfiny [@Habera:aa] for parallel scalability.

Our solver's API receives an abstract energy functional, a user-friendly description of the state of the system **as a dictionary (u, alpha), whre the first elemnt is associated to the reversible field and the second to the irreversible component**, ~~its~~ **the** associated constraints **on the latter**, and the solver's parameters **(an example in the Appendix)**. Solvers can be instantiated calling
```
solver = {Hybrid,Bifurcation,Stability}Solver(
      E,              # An energy (dolfinx.fem.form) 
      state,          # A dictionary of fields describing the system
      bcs,            # A list of boundary conditions
      [bounds],       # A list of bounds (upper and lower) for the state 
      parameters)     # A dictionary of numerical parameters
```
where `[bounds]` are required for the `HybridSolver`, and used calling `solver.solve(<args>)` which triggers the solution of the corresponding variational problem. Here, `<args>` depend on the solver (see the documentation for details).

`HybridSolver` solves a (first order) constrained nonlinear variational inequality, implementing a two-phase hybrid strategy which is _ad hoc_ for energy models typical of applications in damage and fracture mechanics. The first phase (iterative alternate minimisation) is based on a de-facto _industry standard_, conceived to exploit the (partial, directional) convexity of the underlying mechanical models [@bourdin:2000-numerical]. Once an approximate-solution enters the attraction set around a critical point, the solver switches to perform a fully nonlinear step solving a block-matrix problem via Newton's method. This guarantees a precise estimation of the convergence of the first-order nonlinear problem based on the norm of the (constrained) residual. 

`BifurcationSolver` is a variational eigenvalue solver which uses SLEPc.EPS to explore the lower part of the spectrum of the Hessian of the energy, automatically computed performing two directional derivatives. Constraints are accounted for by projecting the full Hessian onto the subspace of inactive constraints [@jorge-nocedal:1999-numerical]. The relevance of this approach is typical of systems with threshold laws. Thus, the `solve` method returns a boolean value indicating whether the restricted Hessian is positive definite. Internally, the solver stores the lower part of the operators' spectrum as an array. 

`StabilitySolver` solves a constrained variational eigenvalue inequality in a convex cone, to check whether the (restricted) nonlinear Hessian operator is positive therein. Starting from an initial guess $z_0^*$, it iteratively computes (eigenvalue, eigenvector) pairs $(\lambda_k, z_k)$ converging to a limit $(\lambda^*, z^*)$ (as $k\to \infty$), by implementing a simple projection and scaling algorithm [@moreau:1962-decomposition], [@pinto-da-costa:2010-cone-constrained].
The positivity of $\lambda^*$ (the smallest eigenvalue) allows to conclude on the stability of the current state (or lack thereof), hence effectively solving P(0). Notice that, if the current state is unstable ($\lambda^*<0$), the minimal eigenmode indicates the direction of energy decrease.

We dedicate a separate contribution to illustrate how the three solvers are algorithmically combined to solve problem P(0) in the case of fracture. \autoref{fig:convergence} illustrates the numerical convergence properties of the `StabilitySolver` in the 1d verification test.

![Rate of convergence for `StabilitySolver` in 1d (cf. benchmark problem below). Targets are the eigenvalue $\lim_k \lambda_k =: \lambda^*$ (pink) and the associated eigen-vector $x^*$ (error curve in blue). Note that the residual vector (green) for the cone problem need not be zero at a solution.\label{fig:convergence}](media/test_1d_stability-spa.pdf)


## Verification

We benchmark our solvers against a nontrivial 1d problem (cf. `test/test_rayleigh.py` in the code repository), namely we compute
$$
\min_{X_0} \mathcal R(z) \quad \text{and} \quad \min_{\mathcal K^+_0} \mathcal R(z) \qquad\qquad [2],$$
**where $z = (v, \beta)$ in $X_0$ and $\mathcal K^+_0$,** using `BifurcationSolver` and `StabilitySolver`, **respectively**. The quantity $\mathcal R(z)$ is a Rayleigh ratio, often used in structural mechanics as a dimensionless global quantity (an energetic ratio of elastic and fracture energies) which provides insight into the stability and critical loading conditions for a structure. For definiteness, using the Sobolev spaces which are the natural setting for second order PDE problems, we set $X_0 = H^1_0(0, 1) \times H^1(0, 1)$ and $\mathcal K^+_0 = H^1_0(0, 1) \times \{\beta \in H^1(0, 1), \beta \geq 0\}$. Let
$$\mathcal R(z):= \dfrac{\int_0^1 a(\beta'(x))^2dx+\int_0^1 b(v'(x) -c\beta(x))^2dx}{\int_0^1\beta(x)^2dx},\qquad\qquad [3]$$ 
 where $a, b, c$ are real coefficients such that $a>0, b>0, c\neq 0$. The quantity above occurs in the stability analysis of a 1d damageable bar, where $b$ is related to the spring constant of the material while $a, c$ encapsulate material, loading, and model parameters, cf. the Appendix of [@pham:2011-the-issues].

**\autoref{fig:profile}-\autoref{fig:phase_diag_cone} below provide a visual representation of the accuracy of the solvers in the solution of problems [2.1] and [2.2] by comparison to analytic solutions.** 
\autoref{fig:phase_diag_D} **represents the size of the support of the mininimiser** ~~compare numerical results with the analytic solution~~, whereas \autoref{fig:phase_diag_ball} and \autoref{fig:phase_diag_cone} show the relative error on the minimum in the space of parameters. **In particular, the phase diagrams \autoref{fig:phase_diag_D}-\autoref{fig:phase_diag_cone} show the solver's precision parametrically with respect to $a, b, c$, uniformly randomly distributed in their corresponding domains. The contour lines are level curves for the solutions (for the numerical values $D^\#$  and  $R^\#$, see eqs. in Appendix), the color maps indicate their magnitude. The red lines highlight the critical threshold that separates trivial (constant) solutions (with $D^*=1$) from nontrivial ones (with $D^*<1$). Colours of data points encode the value of $D^\#$  and  $R^\#$, error bars represent the relative error, and for some (randomly picked) elements we show the error in percentage.** 

![Comparison between profiles of solutions $\beta(x)$ in $X_0$ (left) vs. $\mathcal K^+_0$ (right). In the latter case, the solution $\beta(x)$ has support of size $D\in [0, 1]$.\label{fig:profile}](media/profile_comparison.pdf)

![The size of the support $D$ for the minimiser in the cone. Error bars indicate the absolute error. We observe a clear separation between constant solutions with $D=1$ vs. nontrivial solutions. Where error bars are not shown, the error is none.\label{fig:phase_diag_D}](media/phase_diagram.pdf)


![Minimum value of $\mathcal R$ in $X_0$, numerical computation vs. closed form result. Notice the separation between constant solutions and nontrivial solutions.\label{fig:phase_diag_ball}](media/phase_diagram_R_ball.pdf)

![Minimum value of $\mathcal R$ in $\mathcal K^+_0$, numerical computation vs. closed form results. The outlier $(\pi^2a, bc^2)\sim (4, 0)$ represents a computation which did not reach convergence. The mechanical interpretation is that only states with $R>1$ are energetically stable.\label{fig:phase_diag_cone}](media/phase_diagram_R_cone.pdf)


## Acknowledgements

ALB acknowledges the students of MEC647 (Complex Crack Propagation in Brittle Materials) of the `Modélisation Multiphysique Multiéchelle des Matériaux et des Structures` master program at ENSTA Paris Tech/École Polytechnique for their contributions, motivation, and feedback; Yves Capdeboscq, Jean-Jacques Marigo, Sebastien Neukirch, and Luc Nguyen, for constructive discussions and one key insight that was crucial for this project.
<PLC>The work of PC was supported by the JSPS Innovative Area grant JP21H00102 and  JSPS Grant-in-Aid for Scientific Research (C) JP24K06797. PC holds an honorary appointment at La Trobe University and is a member of GNAMPA.  
</PLC>

## **Appendix**

### **Analytic solutions**
**Given $\mathcal R(z)$ as in [3], the solutions to the minimum problems [2.1] and [2.2] are**
**$$
\min_{X_0} \mathcal{R}(z) = \min\{bc^2, \pi^2 a\}, \quad \text{and} \quad \min_{\mathcal{K}^+_0} \mathcal{R}(z)= \left\{ 
\begin{aligned}
    & bc^2, & \text{if }\pi^2 a \geq bc^2 \\
    & \left(\pi^2 a\right)^{1 / 3}\left(b c^2\right)^{2 / 3}, & \text{if }\pi^2 a < bc^2
\end{aligned}
\right.
$$**
**For details on the computation, cf. [@pham:2011-the-issues]. The associated eigenspace (the minimiser) is, for [2.1], $z^*=(v^*, \beta^*)$ given by**
**
$$
\beta^*(x)=C+A \cos \pi x,\quad \text{ and }\quad v^*(x)=\frac{{c} A}{\pi} \sin \pi x,
$$
where $A=0$ if ${bc}^2<\pi^2 {a}, C=0$ if ${bc}^2>\pi^2 {a}$. $A$ and $C$ are arbitrary real numbers otherwise.
**
**The minimiser $(v^*, \beta^*)$ for [2.2] is given by $v^*$ as above and**
**1. $\beta^*(x)=C>0$, if $\pi^2 {a}>{bc}^2$. **
**2.  $\beta^*(x)=C+A \cos (\pi x)$ with $C>0$ and $|A| \leq C$, if $\pi^2 {a}={bc}^2$.**
**3.
$$
\beta^*(x)=\left\{\begin{array}{ll}
C\left(1+\cos (\pi \frac{x}{{D}})\right) & \text { if } x \in(0, {D}) \\
0 & \text { otherwise }
\end{array} \text { and } \quad \tilde{\beta}^*(x)=\beta_*(1-x),\right.
$$**
**
if $\pi^2 a<b c^2$, where $C$ is an arbitrary positive constant and $D^3=\pi^2 a / b c^2$.
**

### **Numerical parameters**
**We provide an example of the list of numerical parameters associated to the simulation reported in the paper. The list contains all relevant parameters, including geometry, loading (if it applies), and solvers configuration. The rationale is to ensure reproducibility of numerical simulations and clarity in collecting the computational metadata.
**

```
geometry:
  Lx: 1.0
  Ly: 0.1
  N: 50
  geom_type: traction-bar
  geometric_dimension: 2
  lc: 0.02
  mesh_size_factor: 4
model:
  a: 1
  b: 4
  c: 8
  model_dimension: 1
  model_type: 1D
stability:
  checkstability: 'True'
  cone:
    cone_atol: 1.0e-06
    cone_max_it: 400000
    cone_rtol: 1.0e-06
    maxmodes: 3
    scaling: 0.001
  cont_rtol: 1.0e-10
  continuation: 'False'
  eigen:
    eig_rtol: 1.0e-08
    eps_max_it: 100
    eps_tol: 1.0e-05
    eps_type: krylovschur
  inactiveset_gatol: 0.1
  inactiveset_pwtol: 1.0e-06
  inertia:
    ksp_type: preonly
    mat_mumps_icntl_13: 1
    mat_mumps_icntl_24: 1
    pc_factor_mat_solver_type: mumps
    pc_type: cholesky
  is_elastic_tol: 1.0e-06
  linesearch:
    method: min
    order: 4
  maxmodes: 10
  order: 3
solvers:
  damage:
    prefix: damage
    snes:
      ksp_type: preonly
      pc_factor_mat_solver_type: mumps
      pc_type: lu
      snes_atol: 1.0e-08
      snes_linesearch_type: basic
      snes_max_it: 50
      snes_monitor: ''
      snes_rtol: 1.0e-08
      snes_type: vinewtonrsls
    type: SNES
  damage_elasticity:
    alpha_rtol: 0.0001
    criterion: alpha_H1
    max_it: 200
  elasticity:
    prefix: elasticity
    snes:
      ksp_type: preonly
      pc_factor_mat_solver_type: mumps
      pc_type: lu
      snes_atol: 1e-8
      snes_max_it: 200
      snes_monitor: ''
      snes_rtol: 1e-8
      snes_stol: 1e-8
      snes_type: newtontr
  newton:
    linesearch_damping: 0.5
    snes_atol: 1.0e-08
    snes_linesearch_type: basic
    snes_max_it: 30
    snes_monitor: ''
    snes_rtol: 1.0e-08
    snes_type: vinewtonrsls
```


## References

<!-- to compile: docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara -->