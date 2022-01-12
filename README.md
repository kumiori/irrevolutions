# MEC647, Complex Crack Propagation in Briittle Materials
## 2021/22 Ed., Master ENSTA Paris Tech, Polytechnique

From little or nothing, to experimental verification of a complex fracture experiment.
To understand fracture, simple ingredients: one energetic evolution law, one irreversibility constraint, and a whole class of elliptic models.

This repository contains all course material to solving fracture problems arising in solid mechanics with
the variational approach. 
Contents include: scripts and material data.
The computational environment is based on DOLFINx. 

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

Feature branch workflow
https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html

### Weekly updates (merge from main)
```
git checkout main
git pull
git checkout yourname-branch
git merge main
```

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


### Instructors/Authors

- V Lazarus, Sorbonnes Université.
- A Leon Baldelli, CR CNRS.

### Acknowledgements

Corrado Maurini, Sorbonne Université

### License

MIT License, see `LICENSE` file.
