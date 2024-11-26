---
title: Addedndum to 'Variational Solvers for Irreversible Evolutionary Systems'
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
      equal-contrib: false
      affiliation: 2
affiliations:
    - name: $\partial$'Alembert Institute, CNRS, Sorbonne Universités, Place Jussieu 75252 Paris Cedex 05, France
      index: 1
    - name: Institute of Mathematics for Industry, Kyushu University, 744 Motooka, Nishi-ku, Fukuoka, 819-0395, Japan.
      index: 2
date: 1 August 2024
bibliography: paper.bib
geometry: 'left=1.9cm,right=1.9cm,top=1.9cm,bottom=1.9cm'
output: pdf_document
---

## Verification

We benchmark our solvers against a nontrivial 1d problem (cf. `test/test_rayleigh.py` in the code repository), namely we compute

$$
\begin{aligned}
&\min_{X_0} \mathcal R(z) \qquad \qquad [2.1] \\
&\min_{\mathcal K^+_0} \mathcal R(z) \qquad\qquad [2.2]
\end{aligned}
$$

where $z = (v, \beta)$ in $X_0$ and $\mathcal K^+_0$, using `BifurcationSolver` and `StabilitySolver`, respectively. The quantity $\mathcal R(z)$ is a Rayleigh ratio, often used in structural mechanics as a dimensionless global quantity (an energetic ratio of elastic and fracture energies) which provides insight into the stability and critical loading conditions for a structure. For definiteness, using the Sobolev spaces which are the natural setting for second order PDE problems, we set $X_0 = H^1_0(0, 1) \times H^1(0, 1)$ and $\mathcal K^+_0 = H^1_0(0, 1) \times \{\beta \in H^1(0, 1), \beta \geq 0\}$. Let
$$\mathcal R(z):= \dfrac{\int_0^1 a(\beta'(x))^2dx+\int_0^1 b(v'(x) -c\beta(x))^2dx}{\int_0^1\beta(x)^2dx},\qquad\qquad [3]$$
where $a, b, c$ are real coefficients such that $a>0, b>0, c\neq 0$. The quantity above occurs in the stability analysis of a 1d damageable bar, where $b$ is related to the spring constant of the material while $a, c$ encapsulate material, loading, and model parameters, cf. the Appendix of [@pham:2011-the-issues].

\autoref{fig:profile}-\autoref{fig:phase_diag_cone} below provide a visual representation of the accuracy of the solvers in the solution of problems [2.1] and [2.2] by comparison to analytic solutions.
\autoref{fig:phase_diag_D} represents the size of the support of the mininimiser, whereas \autoref{fig:phase_diag_ball} and \autoref{fig:phase_diag_cone} show the relative error on the minimum in the space of parameters. In particular, the phase diagrams \autoref{fig:phase_diag_D}-\autoref{fig:phase_diag_cone} show the solver's precision parametrically with respect to $a, b, c$, uniformly randomly distributed in their corresponding domains. The contour lines are level curves for the solutions (for the numerical values $D^\#$ and $R^\#$, see eqs. in Appendix), the color maps indicate their magnitude. The red lines highlight the critical threshold that separates trivial (constant) solutions (with $D^*=1$) from nontrivial ones (with $D^*<1$). Colours of data points encode the value of $D^\#$ and $R^\#$, error bars represent the relative error, and for some (randomly picked) elements we show the error in percentage.

![Comparison between profiles of solutions $\beta(x)$ in $X_0$ (left) vs. $\mathcal K^+_0$ (right). In the latter case, the solution $\beta(x)$ has support of size $D\in [0, 1]$.\label{fig:profile}](media/profile_comparison.pdf)

![The size of the support $D$ for the minimiser in the cone. Error bars indicate the absolute error. We observe a clear separation between constant solutions with $D=1$ vs. nontrivial solutions. Where error bars are not shown, the error is none.\label{fig:phase_diag_D}](media/phase_diagram.pdf)

![Minimum value of $\mathcal R$ in $X_0$, numerical computation vs. closed form result. Notice the separation between constant solutions and nontrivial solutions.\label{fig:phase_diag_ball}](media/phase_diagram_R_ball.pdf)

![Minimum value of $\mathcal R$ in $\mathcal K^+_0$, numerical computation vs. closed form results. The outlier $(\pi^2a, bc^2)\sim (4, 0)$ represents a computation which did not reach convergence. The mechanical interpretation is that only states with $R>1$ are energetically stable.\label{fig:phase_diag_cone}](media/phase_diagram_R_cone.pdf)

## Acknowledgements

ALB acknowledges the students of MEC647 (Complex Crack Propagation in Brittle Materials) of the `Modélisation Multiphysique Multiéchelle des Matériaux et des Structures` master program at ENSTA Paris Tech/École Polytechnique for their contributions, motivation, and feedback; Yves Capdeboscq, Jean-Jacques Marigo, Sebastien Neukirch, and Luc Nguyen, for constructive discussions and one key insight that was crucial for this project.
<PLC>The work of PC was supported by the JSPS Innovative Area grant JP21H00102 and JSPS Grant-in-Aid for Scientific Research (C) JP24K06797. PC holds an honorary appointment at La Trobe University and is a member of GNAMPA.  
</PLC>

## **Appendix**

### **Analytic solutions**

Given $\mathcal R(z)$ as in [3], the solutions to the minimum problems [2.1] and [2.2] are

$$
\min_{X_0} \mathcal{R}(z) = \min\{bc^2, \pi^2 a\}, \quad \text{and} \quad \min_{\mathcal{K}^+_0} \mathcal{R}(z)= \left\{
\begin{aligned}
    & bc^2, & \text{if }\pi^2 a \geq bc^2 \\
    & \left(\pi^2 a\right)^{1 / 3}\left(b c^2\right)^{2 / 3}, & \text{if }\pi^2 a < bc^2
\end{aligned}
\right.
$$

For details on the computation, cf. [@pham:2011-the-issues]. The associated eigenspace (the minimiser) for [2.1] is $z^*=(v^*, \beta^*)$ given by

$$
\beta^*(x)=C+A \cos \pi x,\quad \text{ and }\quad v^*(x)=\frac{{c} A}{\pi} \sin \pi x,
$$

where $A=0$ if ${bc}^2<\pi^2 {a}, C=0$ if ${bc}^2>\pi^2 {a}$. $A$ and $C$ are arbitrary real numbers otherwise.
The minimiser $(v^*, \beta^*)$ for [2.2] is given by $v^*$ as above and

$$
\beta^*(x)=\left\{
\begin{aligned}
C,\qquad & \text{ if }\pi^2 {a}>{bc}^2 \\
C+A \cos (\pi x),\qquad & \text{ if } \pi^2 {a}={bc}^2 \text{, with }C>0 \text{ and }|A| \leq C\\
C\left(1+\cos (\pi \frac{x}{{D}})\right), \qquad & \text{ if } \pi^2 {a}<{bc}^2,  \text { for } x \in(0, {D})
\end{aligned}\right.
$$

where $C$ is an arbitrary positive constant and $D^3=\pi^2 a / b c^2$; with the understanding that in the third case ($\pi^2 {a}<{bc}$), $\beta^*(x)=0$ for $x\notin (0, D)$ and $\tilde \beta^*(x):=\beta^*(1-x)$ is also a solution.

### **Numerical parameters**

We provide an example of the list of numerical parameters associated to the simulation reported in the paper. The list contains all relevant parameters, including geometry, loading (if it applies), and solvers configuration. The rationale is to ensure reproducibility of numerical simulations and clarity in collecting the computational metadata.

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
