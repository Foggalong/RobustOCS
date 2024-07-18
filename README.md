# RobustOCS

Tools for solving robust optimal contribution selection problems in Python. All code and examples in RobustOCS are fully and freely available for all use under the MIT License.

## Dependencies

It depends on Python 3.10+, using [NumPy](https://pypi.org/project/numpy) for linear algebra and [SciPy](https://scipy.org) for sparse matrix objects. As a solver it can either use:

- [Gurobi](https://www.gurobi.com) (commercial) via [gurobipy](https://pypi.org/project/gurobipy),
- [HiGHS](https://highs.dev) (free software) via [highspy](https://pypi.org/project/highspy).

**NOTE**: Gurobi don't yet have [NumPy v2 support]. By extension, this module will continue to use NumPy 1.2x only until gurobipy is updated.

[NumPy v2 support]: https://support.gurobi.com/hc/en-us/articles/25787048531601-Compatibility-issues-with-numpy-2-0

## Examples

The [GitHub wiki] includes documentation written by which explains the usage and parameters in more detail, alongside some worked examples (the code for which is in [`examples/`](examples/)). This includes a realistic simulated example from [Gregor Gorjanc] and [Ivan Pocrnić].

[GitHub wiki]: https://github.com/Foggalong/RobustOCS/wiki
[Gregor Gorjanc]: https://www.ed.ac.uk/profile/gregor-gorjanc
[Ivan Pocrnić]: https://www.ed.ac.uk/profile/ivan-pocrnic
