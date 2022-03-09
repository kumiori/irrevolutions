# MEC647, Complex Crack Propagation in Brittle Materials
## 2021/22 Ed., Master ENSTA Paris Tech, Polytechnique

From little or nothing, to experimental verification of a complex fracture experiment.
To understand fracture, simple ingredients: one energetic evolution law, one irreversibility constraint, and a whole class of elliptic models.

This repository contains all course material to solving fracture problems arising in solid mechanics with
the variational approach. 
Contents include: scripts and material data.
The computational environment is based on DOLFINx. 

## The Result log(03/22)

We like it, rough - for now.\
Cf. below and see beyond.

What do we show? \
Pictures in colour: in Practice \
showing matching of two extremes. \
Theory and reality, through coumputation \
and analysis, proofs and concepts \
all hidden, explicitly, in few pixels.\
Complex patterns, interacting fields, \
real energy, in evolution with no \
way back.

MEC647/22 has been the first compact \
set to attack a validation task against \
down-to-Earth lab experiments. The \
result, in practice, is Major. 

Matching has never been a matter of luck, \
nor of hope, nor of chance. For us, \
it's been matter of understanding.

What do we hide? What do we see? \
The connection between singularities and cracks, \
in the strict sense, in the social sphere, as a \
function of time.

Good enough, in eaten form.

_This is a trace._
/CR 83252

1) Universal Shapes Formed by Two Interacting Cracks, 10.1103/PhysRevLett.105.125505
2) Rough computation, codename "en-passant", in practice/ check the scripts
3) Singular chouquettes - what's the link?, Food Lab @base 

<img width="250" align="left" alt="Screen Shot 2022-03-04 at 10 03 44 AM" src="https://user-images.githubusercontent.com/2798610/156734844-ac56dec7-5689-454d-acca-10ca8392b204.png">
<img width="270" align="left" alt="Screen Shot 2022-03-04 at 10 03 44 AM" src="https://user-images.githubusercontent.com/2798610/156734995-cac46287-2b2d-42f2-8338-0ca9800abd37.png">
<img width="250" alt="Screen Shot 2022-03-04 at 10 15 39 AM" src="https://user-images.githubusercontent.com/2798610/156734790-8db34e2a-6a28-4314-bc64-a187b34a6ae9.png">


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



### Weekly updates (merge from main)
```
git checkout main
git pull
git checkout yourname-branch
git merge main
```
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
