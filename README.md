# RobustOCS

[![PyPi](https://img.shields.io/pypi/v/robustocs.svg)](https://pypi.python.org/pypi/robustocs) [![Check Build](https://github.com/Foggalong/RobustOCS/actions/workflows/check-build.yml/badge.svg?branch=main)](https://github.com/Foggalong/RobustOCS/actions/workflows/check-build.yml)

Tools for solving robust optimal contribution selection problems in Python. All code and examples in RobustOCS are fully and freely available for all use under the MIT License.

## Installation

The latest release can be installed from the [PyPI repository](https://pypi.org/project/robustocs/) using

```bash
pip install robustocs
```

Alternatively, it use the latest (potentially unstable) version from GitHub, use

```bash
git clone https://github.com/Foggalong/RobustOCS.git
pip install RobustOCS/
```

Either way, the package depends on Python 3.10+, using [NumPy](https://pypi.org/project/numpy) for linear algebra and [SciPy](https://scipy.org) for sparse matrix objects. As a solver it can either use [Gurobi](https://www.gurobi.com) (commercial) via [gurobipy](https://pypi.org/project/gurobipy) or [HiGHS](https://highs.dev) (free software) via [highspy](https://pypi.org/project/highspy).

## Quick-start

Suppose we have a breeding cohort whose relationships are modelled some matrix $\Sigma$, and whose expected breeding values have mean $\bar{\boldsymbol\mu}$ and variance $\Omega$. If these are [saved in files](https://github.com/Foggalong/RobustOCS/wiki/File-Formats), finding the robust optimal contribution selection is as simple as opening Python and running

```python
import robustocs as rocs

selection, objective, merit, coancestry = rocs.solveROCS(
    sigma_filename="cohort-relationships.txt",
    mu_filename="breeding-means.txt",
    omega_filename="breeding-variances.txt",
    sex_filename="cohort-sexes.txt",
    method='robust', lam=0.5, kappa=1
)
```

where `selection` will be an array of optimal selections, `objective` is a score of the selection, `merit` the expected genetic merit, and `coancestry` the group co-ancestry.

## Documentation

The [GitHub wiki] includes documentation written by which explains the usage and parameters in more detail, alongside some worked examples using the more granular solver functions (the data for which is in [`examples/`](examples/)). This includes a realistic simulated example from [Gregor Gorjanc] and [Ivan Pocrnić].

[GitHub wiki]: https://github.com/Foggalong/RobustOCS/wiki
[Gregor Gorjanc]: https://www.ed.ac.uk/profile/gregor-gorjanc
[Ivan Pocrnić]: https://www.ed.ac.uk/profile/ivan-pocrnic
