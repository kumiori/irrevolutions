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
 - name: $\partial$'Alembert Institute, CNRS, Sorbonne Université, Place Jussieu 75252 Paris Cedex 05, France
   index: 1
 - name: Institute of Mathematics for Industry, Kyushu University, 744 Motooka, Nishi-ku, Fukuoka, 819-0395, Japan.
   index: 2
date: 1 August 2024
bibliography: paper.bib

---

# Summary

We study irreversible evolutionary processes with a general energetic notion of stability. With this contribution, we release three nonlinear variational solvers as modular components (based on FEniCSx/dolfinx) that address three mathematical optimisation problems. They are general enough to apply, in principle, to evolutionary systems with instabilities, jumps, and emergence of patterns. Systems with these qualities are commonplace in diverse arenas spanning from quantum to continuum mechanics, economy, social sciences, and ecology. Our motivation proceeds from fracture mechanics, with the ultimate goal of deploying a transparent numerical platform for scientific validation and prediction of large scale natural fracture phenomena. Our solvers are used to compute _one_ solution to a problem encoded in a system of two inequalities: one (pointwise almost-everywhere) constraint of irreversibility and one global energy statement.

# Statement of need

Quasi-static evolution problems arising in fracture are strongly nonlinear [@marigo:2023-la-mecanique], [@bourdin:2008-the-variational]. They can admit multiple solutions, or none [@leon-baldelli:2021-numerical]. This demands both a functional theoretical framework and practical computational tools for real case scenarios. Due to the lack of uniqueness of solutions, it is fundamental to leverage the full variational structure of the problem and investigate solutions up to second order, to detect nucleation of stable modes and transitions of unstable states. The stability of a multiscale system along its nontrivial evolutionary paths in phase space is a key property that is difficult to check: numerically, for real case scenarios with several length scales involved, and analytically, in the infinite-dimensional setting. Despite the concept of unilateral stability is classical in the variational theory of irreversible systems [@mielke] and the mechanics of fracture [@FRANCFORT] (see also [@bazant, @PETRYK, @Quoc, @Quoc2002]), few studies have explored second-order criteria for crack nucleation and evolution. Although sporadic, these studies are significant, including [@pham:2011-the-issues], [@Pham2013aa], [@SICSIC], [@leon-baldelli:2021-numerical], and [@camilla]. The current literature in computational fracture mechanics predominantly focuses on unilateral first-order criteria, systematically neglecting the exploration of higher-order information for critical points. To the best of our knowledge, no general numerical tools are available to address second-order criteria in evolutionary nonlinear irreversible systems and fracture mechanics.

To fill this gap, our nonlinear solvers offer a flexible toolkit for advanced stability analysis of systems which evolve with constraints.

# **Functionality**

We attack the following abstract problem which encodes a selection principle:

$$
P(0):\text{ Given } T >0, \text{ find an } \text{ irreversible-constrained evolution } y_t
$$

$$
y_t: t\in [0, T]\mapsto X_t
\text{ such that}
$$

$$\text{[Unilateral Stability]} \qquad E(y_t) \leq E(y_t + z), \quad \forall z \in V_0 \times K^+_0\qquad [1]$$

Above, $T$ defines a horizon of events. The system is represented by its total energy $E$ and $X_t$ is the time-dependent space of admissible states. A generic element of $X_t$ contains a macroscopic field that can be externally driven (or controlled, e.g. via boundary conditions) and an internal field (akin to an internal degree of order). In the applications of fracture, the kinematic variable is a vector-valued displacement $u(x)$ and the degree of order $\alpha(x)$ controls the softening of the material. Irreversibility applies to the internal variable, hence an <abbr >irreversible-constrained</abbr> evolution is a mapping parametrised by $t$ such that $\alpha_t(x)$ is non-decreasing with respect to $t$. The kinematic variable is subject to bilateral variations belonging to a linear subset of a Sobolev vector space $V_0$, whereas the test space for the internal order parameter $K^+_0$ only contains positive fields owing to the irreversibility constraint. The main difficulties are to correctly enforce unilateral constraints and to account for the changing nature of the space of variations.

`HybridSolver` (1) `BifurcationSolver,` (2) and `StabilitySolver` (3) address the solution of [1] in three stages:

1. A constrained variational inequality; that is first order necessary conditions for unilateral equilibrium.

2. A singular variational eigen-problem in a vector space; that is
   a bifurcation problem indicating uniqueness (or lack thereof) of the evolution path.
3. A constrained eigen-inequality in a convex cone; originating from a second order eigenvalue problem indicating stabilty of the system (or lack thereof).

These numerical tools can be used to study general evolutionary problems formulated in terms of fully nonlinear functional operators in spaces of high or infinite dimension. In this context, systems can have surprising and complicated behaviours such as symmetry breaking bifurcations, endogenous pattern formation, localisations, and separation of scales. Our solvers can be extended or adapted to a variety of systems described by an energetic principle formulated as in [1].

## Software

Our solvers are written in `Python` and are built on `DOLFINx`, an expressive and performant parallel distributed computing environment for solving partial differential equations using the finite element method [@dolfinx2023preprint]. It enables us wrapping high-level functional mathematical constructs with full flexibility and control of the underlying linear algebra backend. We use PETSc [@petsc-user-ref], petsc4py [@dalcinpazklercosimo2011], SLEPc.EPS [@hernandez:2005-slepc], and dolfiny [@Habera:aa] for parallel scalability.

Our solver's API receives an abstract energy functional, a user-friendly description of the state of the system as a dictionary (u, alpha), where the first element is associated to the reversible field and the second to the irreversible component, the associated constraints on the latter, and the solver's parameters (see an example in the [Addendum](https://doi.org/10.5281/zenodo.14222736)). Solvers can be instantiated calling

```
solver = {Hybrid,Bifurcation,Stability}Solver(
      E,              # An energy (dolfinx.fem.form)
      state,          # A dictionary of fields describing the system
      bcs,            # A list of boundary conditions
      [bounds],       # A list of bounds (lower and upper) for the order parameter
      parameters     # A dictionary of numerical parameters
      )
```

where `[bounds]=[lower, upper]` are required for the `HybridSolver`. Calling `solver.solve(<args>)` triggers the solution of the corresponding variational problem. Here, `<args>` depend on the solver (see the documentation for details).

`HybridSolver` solves a (first order) constrained nonlinear variational inequality, implementing a two-phase hybrid strategy which is _ad hoc_ for energy models typical of applications in damage and fracture mechanics. The first phase (iterative alternate minimisation) is based on a de-facto industry standard, conceived to exploit the (partial, directional) convexity of the underlying mechanical models [@bourdin:2000-numerical]. Once an approximate-solution enters the attraction set around a critical point, the solver switches to perform a fully nonlinear step solving a block-matrix problem via Newton's method. This guarantees a precise estimation of the convergence of the first-order nonlinear problem based on the norm of the (constrained) residual.

`BifurcationSolver` is a variational eigenvalue solver which uses SLEPc.EPS to explore the lower part of the spectrum of the Hessian of the energy, automatically computed performing two directional derivatives. Constraints are accounted for by projecting the full Hessian onto the subspace of inactive constraints [@jorge-nocedal:1999-numerical]. The relevance of this approach is typical of systems with threshold laws. Thus, the `solve` method returns a boolean value indicating whether the restricted Hessian is positive definite. Internally, the solver stores the lower part of the operators' spectrum as an array.

`StabilitySolver` solves a constrained variational eigenvalue inequality in a convex cone, to check whether the (restricted) nonlinear Hessian operator is positive therein. Starting from an initial guess $z_0^*$, it iteratively computes (eigenvalue, eigenvector) pairs $(\lambda_k, z_k)$ converging to a limit $(\lambda^*, z^*)$ (as $k\to \infty$), by implementing a simple projection and scaling algorithm [@moreau:1962-decomposition], [@pinto-da-costa:2010-cone-constrained].
The positivity of $\lambda^*$ (the smallest eigenvalue) allows to conclude on the stability of the current state (or lack thereof), hence effectively solving P(0). Notice that, if the current state is unstable ($\lambda^*<0$), the minimal eigenmode indicates the direction of energy decrease.

We dedicate a separate contribution to illustrate how the three solvers are algorithmically combined to solve problem P(0) in the case of fracture. \autoref{fig:convergence} illustrates the numerical convergence properties of the `StabilitySolver` in a 1d verification test.

In a [supplementary document](https://doi.org/10.5281/zenodo.14222736), we perform a thorough verification of the code through parametric benchmark for investigating the stability of a 1D mechanical system, providing analytical expressions used for comparison with numerical solutions, as well as all parameters (numerical and physical) employed in the calculations.

<!-- In conclusion, we also provide a separate appendix that includes the . -->

![Rate of convergence for `StabilitySolver` in 1d (cf. benchmark problem below). Targets are the eigenvalue $\lim_k \lambda_k =: \lambda^*$ (pink) and the associated eigen-vector $x^*$ (error curve in blue). Note that the residual vector (green) for the cone problem need not be zero at a solution.\label{fig:convergence}](media/test_1d_stability-spa.pdf)

## Acknowledgements

ALB acknowledges the students of MEC647 (Complex Crack Propagation in Brittle Materials) of the `Modélisation Multiphysique Multiéchelle des Matériaux et des Structures` master program at ENSTA Paris Tech/École Polytechnique for their contributions, motivation, and feedback; Yves Capdeboscq, Jean-Jacques Marigo, Sebastien Neukirch, and Luc Nguyen, for constructive discussions and one key insight that was crucial for this project.
<PLC>The work of PC was supported by the JSPS Innovative Area grant JP21H00102 and JSPS Grant-in-Aid for Scientific Research (C) JP24K06797. PC holds an honorary appointment at La Trobe University and is a member of GNAMPA.  
</PLC>

## References

<!-- to compile: docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara -->
