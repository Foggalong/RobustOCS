# RobustOCS

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

## Examples

The [GitHub wiki] includes documentation written by which explains the usage and parameters in more detail, alongside some worked examples (the data for which is in [`examples/`](examples/)). This includes a realistic simulated example from [Gregor Gorjanc] and [Ivan Pocrnić].

[GitHub wiki]: https://github.com/Foggalong/RobustOCS/wiki
[Gregor Gorjanc]: https://www.ed.ac.uk/profile/gregor-gorjanc
[Ivan Pocrnić]: https://www.ed.ac.uk/profile/ivan-pocrnic
