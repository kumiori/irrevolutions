# Crack Propagation in Brittle Materials

## 2024
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06897/status.svg)](https://doi.org/10.21105/joss.06897)
[![Run Tests in Docker Container](https://github.com/kumiori/irrevolutions/actions/workflows/workflow.yaml/badge.svg)](https://github.com/kumiori/irrevolutions/actions/workflows/workflow.yaml)
[![Test Conda Installation](https://github.com/kumiori/irrevolutions/actions/workflows/conda.yml/badge.svg)](https://github.com/kumiori/irrevolutions/actions/workflows/conda.yml)
[![Test Ubuntu Installation](https://github.com/kumiori/irrevolutions/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/kumiori/irrevolutions/actions/workflows/ubuntu.yml)

`Irrevolutions` is a computational stability analysis toolkit designed to solve nonlinear and nonconvex evolutionary problems using advanced numerical methods. It provides efficient algorithms for computing solutions for constrained minimum problems with application to irreversible evolutions (hence its name). In particular, this framework is relevant in the context of fracture and damage mechanics.

**Irreversible Evolution of Damage**

Let $y=(\alpha, u)$ be an admissible state of a brittle system where $\alpha: \Omega \mapsto [0, 1]$ is a smooth damage field which identifies cracks (where $\alpha =1$) and $u$ is a displacement field. Provided a material model (an energy functional) $E_\ell$, given a time horizon $T$, let's find a map $t \in [0, T]\mapsto y_t$ such that: damage is non-decreasing and the observed state $y_t$ is energy-minimal, among admissible variations.

## How to contribute

### Reporting bugs

If you find a bug in `irrevolutions`, please report it on the GitHub issue tracker.

## Suggesting enhancements

If you wish to suggest a new feature or an improvement of a current feature, you can submit this on the issue tracker.

## Contributing code (submitting a pull request)

To contribute code `irrevolutions`, create a pull request. If you want to contribute, but are unsure where to start, get in touch with the authors.

On opening a pull request, unit tests will run on GitHub Continuous Integration. You can click on these in the pull request to see where (if anywhere) the tests are failing.

For more details on the pull request workflow, check
https://docs.godotengine.org/en/3.1/community/contributing/pr_workflow.html

### Installation

Before installing `irrevolutions`, ensure you have `dolfinx` and other dependencies installed.

DOLFINx (and some other libraries like MPI and PyVista) have complex dependencies, it may be best to install them using conda from the conda-forge channel. Conda handles the environment setup and dependency conflicts well for these low-level libraries. Otherwise, you install from source using Spack or us a pre-built docker image.

Then, install your Python package dependencies with Poetry. After installing system-level dependencies with Conda (or another method), you can use Poetry to manage the Python-specific packages and virtual environment for your project. Poetry will not attempt to reinstall DOLFINx or other system-level packages.

More specifically, you can install `dolfinx` using one of the following methods:

- Using conda

```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx=0.9 mpich pyvista sympy pandas pyyaml
conda activate fenicsx-env
```

- Using Spack
  see https://github.com/FEniCS/dolfinx/blob/main/README.md#spack

- Using Apt (ubuntu 23.04 build)

```bash
apt-get install -y  software-properties-common python3-pip git libgl1-mesa-glx xvfb libglu1 libxcursor1 libxinerama1

add-apt-repository ppa:fenics-packages/fenics
apt update
apt-install fenicsx
```

For detailed instructions, see https://github.com/FEniCS/dolfinx/blob/main/README.md#installation

- Using a Docker container

For an ARM-based machine:

```bash
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:stable
```

For an AMD64 machine:

```bash
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:stable-amd64
```

For a windows box:

```bash
docker run --rm -ti -v "C:/...":/home/numerix" -w /home/numerix kumiori3\numerix:stable-amd64
```

Clone this repository:

```bash
git clone https://github.com/kumiori/irrevolutions.gt
cd irrevolutions
```

Finally, `irrevolutions` can be installed using setuptools from the root directory

```bash
python3 -m pip install .
```

---

This code was initially conceived as a support for the teaching course MEC647,
(Complex) Crack Propagation in Brittle Materials, delivered to the students of the international master programme, joint between École Polytechnique and ENSTA-Paristech throughout 2020-2022.

### Acknowledgements

To all the students for their effort, participation, and motivation.

This project contains code from the DOLFINy project (https://github.com/fenics-dolfiny/dolfiny), which is licensed under the LGPLv3 license. We acknowledge and thank the DOLFINy contributors for their work.

See paper.md

### License

See `LICENSE` file.

Each file should have at least the "copyright" line and a pointer to where the full notice is found.

    <Irrevolutions is scientific software, it is conceived to compute evolution paths upon a general notion of (unilateral) stability. It applies to fracture and, maybe, not only.>

    Copyright or copyLeft (C) <~0>  <ALB/83252>

    This program is free software. Here, the term 'free' has all to do
    with freedom and nothing to do with price. You can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed without hope that it will be useful,
    WITHOUT ANY WARRANTY, but with pointers to ONE or SEVERAL PROOFS; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE,
    if-not to compute constrained evolutions of complex systems, whether Natural
    or Social. THIS MATERIAL IS BASED UPON GENERAL RESULTS. See the GNU General
    Public License for more details, see your favourite Functional Analysis reference
    book for further abstraction.

    You should have received a copy of the GNU General Public License
    along with `irrevolution`.  If not, see <https://www.gnu.org/licenses/>.

### Further information

## Star History

<a href="https://star-history.com/#kumiori/irrevolutions&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date" />
 </picture>
</a>
