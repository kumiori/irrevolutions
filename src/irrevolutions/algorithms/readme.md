# Algorithms

`so.py` (standing for Second Order) deals with solving PDEs via the Finite Element Method in Fenics and solving eigenvalue problems. Specifically, it deals with bifurcation problem and our concept of stability, which is encoded and solved as an eigenvalue problem in a cone of directions.

`am.py`
Contains two classes, `AlternateMinimisation` and `HybridSolver`.
 
`AlternateMinimisation` implements an alternate minimization algorithm by solving a system of equations. 
Depending on the specified convergence criteria (e.g., based on error norms vs. residual norms), the solver loop terminates if the specified tolerance is 
reached.
It implements optimality conditions of the first order, via alternate variations.

HybridSolver is a subclass of AlternateMinimisation, indicating it likely inherits some functionality from it.

Ls.py defines a LineSearch class for performing line searches during optimization procedures. It allows to follow bifurcation of states and to bifurcate along multiple solutions.
 
It takes the current state, a perturbation, and an interval as inputs, and returns the optimal step length h_opt. It computes the energy at discretized points within the specified interval and performs polynomial interpolation to find the minimum energy. Is has an option to perform a random search.
The perturb method perturbs the current state based on the computed step length $h$. It updates the state by adding a perturbation $h \star v$ to $u$ and $h \star \beta$ to $\alpha$.
Admissible Interval Method:



The admissible_interval method computes the admissible interval for the line search. It takes into account the perturbation direction and the current state of the system to determine the feasible interval for the step length.
Unilateral Interval Method:

The get_unilateral_interval method computes a unilateral interval for the line search. It considers only positive perturbation directions and computes the maximum step length such that the updated alpha remains within the interval $[0, 1]$.
MPI Parallelization:

The code uses MPI for parallel computing, utilizing functionalities from the mpi4py module. This allows parallelization of operations across multiple processors.
Overall, this class provides functionality for performing line searches during optimization procedures, particularly in the context of finite element simulations.






 
