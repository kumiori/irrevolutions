This folder contains a bunch of tutorials for the resolution of systems displaying bifurcation of multiple solutions.


playground/tutorials/iceice_1Bifurcation.ipynb

We solve an instance of unilaterally evolving system, wildly nonlinear and singularly perturbed [typo, to be fixed], nonconvex hence allowing many minimisers. Expect broken symmetries starting from isotropy, constant coefficients, and homogeneous data.
Compute displacement field (u, scalar variable) and damage field ($\alpha$, scalar variable, typically $0<\alpha<1$) in a bar of length $L$.


$\bullet$	The static problem: Given a load (boundary conditions) and an initial state of damage ($\alpha_0$, the code finds the equilibrium displacement and repartition of damage. This is solved by solving a local minimization problem for a non-convex mechanical energy.

$\bullet$	The evolution problem: Given a load history (boundary conditions as a function of time) and an initial state of damage $\alpha_0$, the code finds the evolution of equilibrium displacement and repartition of damage. This is a time dependent solution (a pair, given by $u\to(u_t,\alpha_t)$ where minimization is performed for irreversible damage (that is, $\alpha_t$ growing in time, the material cannot heal). Due to non-convexity of the problem, multiple solutions may originate depending on the value of critical loads. Here we keep track of the bifurcation of such solutions and follow the path of stable state. The solutions always verify an energy balance (Ext. power) = (Internal energy flux).


playground/tutorials/mec647_BanquiseLune_12.ipynb
TBD

playground/tutorials/mec647_Banquise_11.ipynb
![image](https://github.com/kumiori/irrevolutions/assets/162834058/83328583-1908-4653-9e0c-7db9f08f6105)
TBD


