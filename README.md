# Complex Crack Propagation in Brittle Materials
## 202* Forall T

From little or nothing, to experimental verification of a complex fracture experiment.
We solve the following (difficult) problem:

**Evolution of Damage (in the Cone)**
 
Let $y=(\alpha, u)$ be an admissible state of a brittle system where $\alpha: \Omega \mapsto [0, 1]$ is a smooth damage field which identifies cracks (where $\alpha =1$) and $u$ is a displacement field. Provided a material model (an energy) $E_\ell$, given a time horizon $T$, let's find a map $t \in [0, T]\mapsto y_t$ such that: damage is non-decreasing and the observed state $y_t$ is energy-minimal, among admissible variations. 

## The current branch

We implement a Cone-Recipe. Take it as a surprise cake that opens upon critical conditions.

### Running the notebooks (to be tested *prior* to course start)

To run a Docker container on a local machine.

1. Install Docker following the instructions at
   https://www.docker.com/products/docker-desktop.

2. Clone this repository using git:

       git clone https://github.com/kumiori/mec647.git

3. Run `./launcher.sh`.

Although we recommend using Docker locally, you can also use the cloud-based Google Colab service to execute the notebooks:

### Prerequisites

The course will assume basic knowledge of the theory of infinitesimal elasticity and
finite element methods.

Basic knowledge of Python will be assumed, see https://github.com/jakevdp/WhirlwindTourOfPython
to brush up if you feel unsure.

Basic knowledge of git as a versioning system with feature-branch workflow
https://gist.github.com/brandon1024/14b5f9fcfd982658d01811ee3045ff1e

Remember to set your name and email before pushing to the repository,
either locally or globally, see https://www.phpspiderblog.com/how-to-configure-username-and-email-in-git/

#### Feature branch workflow

For each new feature you wish to implement, create a branch named ```{yourname}-{feature}```, 
as in ```andres-meshes```.

https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html

 - Create your feature branch:`git checkout -b username-feature`
 - To push your branch: `git push -u origin feature_branch_name`
 - Create a pull request on the main branch for merging. Somebody should approve the pull-request. -

### Weekly updates (merge from main)
```
git checkout main
git pull
git checkout yourname-branch
git merge main
```

Asymmetrically, feature-work is `rebased`.

### To run the code (on Docker)

First, run the container, attaching an interactive session and sharing data space 
(the current dir) between the host and the container (the syntax is origin:target).

On a Mac:
```
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:latest
```

On a Windox:
```
docker run --rm -ti -v "C:/...":/home/numerix" -w /home/numerix kumiori3\numerix:latest
```

This code was initially conceived as a support for the teaching course MEC647, 
(Complex) Crack Propagation in Brittle Materials, delivered to the students of the international
master programme, joint between École Polytechnique and ENSTA-Paristech throughout 2020-2022. 
Hence the cryptic repository name.

### Course Schedule


- 0 Introduction, motivation, and a simple experiment, in theory.
- 1 Numerics, from the basics to the solution of a linear variational problem 
- 2 The notion of stability, stability of a brittle bar. The elasticity problem
- 3 Energetics link fracture to damage. Project walkthrough 
- 4 The damage problem: analytics and numerics
- 5 Calibration (material, numerical) and tests
- 6 Data analysis and presentation
- Rest
- 7 Defence


### Authors
- Cf. commit messages
  
### Course Instructors/Authors

- V Lazarus, Sorbonne Université.
- A Leon Baldelli, CR CNRS.


### Acknowledgements

To all the students for their effort, participation, and motivation.

Corrado Maurini, Sorbonne Université.

### License

MIT License, see `LICENSE` file.

### Tests

test_1d.py
test_2d.py

test_cone_convergence.py
test_cone_project.py
test_elasticity.py
test_extend.py
test_rayleigh.py
test_vector_info.py
test_vi.py
test_traction.py
test_bifurcation.py
test_logging_mpi.py
test_scatter.py

