# Crack Propagation in Brittle Materials
## 2024

From little or nothing, to experimental verification of a complex fracture experiment.
We solve the following (difficult) problem:

**Evolution of Damage (irreversible)**
 
Let $y=(\alpha, u)$ be an admissible state of a brittle system where $\alpha: \Omega \mapsto [0, 1]$ is a smooth damage field which identifies cracks (where $\alpha =1$) and $u$ is a displacement field. Provided a material model (an energy) $E_\ell$, given a time horizon $T$, let's find a map $t \in [0, T]\mapsto y_t$ such that: damage is non-decreasing and the observed state $y_t$ is energy-minimal, among admissible variations. 


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

On an ARM-based machine:
```
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:stable
```
On an AMD64 machine:
```
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:stable-amd64
```

On a windows box:
```
docker run --rm -ti -v "C:/...":/home/numerix" -w /home/numerix kumiori3\numerix:stable-amd64
```

To install the software, run DOLFINx through the aforementioned docker container and install the irrevolutions-package with 

```python3 -m pip install .```

from the root of this repository


This code was initially conceived as a support for the teaching course MEC647, 
(Complex) Crack Propagation in Brittle Materials, delivered to the students of the international
master programme, joint between École Polytechnique and ENSTA-Paristech throughout 2020-2022. 
Hence the cryptic repository name.


### Acknowledgements

To all the students for their effort, participation, and motivation.

See paper.md

### License

See `LICENSE` file.

Each file should have at least the "copyright" line and a pointer to where the full notice is found.

    <Irrevolutions is scientific software, it is conceived to compute evolution paths
    upon a general notion of (unilateral) stability. It applies to fracture and, maybe,
    not only.>

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
    along with this program.  If not, see <https://www.gnu.org/licenses/>.



## Star History

<a href="https://star-history.com/#kumiori/irrevolutions&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=kumiori/irrevolutions&type=Date" />
 </picture>
</a>
