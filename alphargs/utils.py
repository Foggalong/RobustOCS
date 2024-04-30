# -*- coding: utf-8 -*-
"""Utilities

This file contains additional utilities, such as for printing and comparing
portfolios produced by the solvers.
"""

import numpy as np


def print_compare_solutions(
    portfolio1,  # : npt.NDArray[np.float64],  # BUG Gurobi typing broken
    portfolio2,  # : npt.NDArray[np.float64],  # BUG Gurobi typing broken
    objective1: float,
    objective2: float,
    precision: int = 5,
    z1: float | None = None,
    z2: float | None = None,
    name1: str = "First",
    name2: str = "Second"
) -> None:
    """
    Takes two solutions (comprised of at least a portfolio and objective value,
    plus an optional z-value and/or solution name) as inputs, and prints a
    comparison of the two solutions to the terminal. The number of decimals
    values are displayed to defaults to 5, but can be changed through the
    precision argument.
    """

    dimension = portfolio1.size
    order = len(str(dimension))

    # HACK header breaks if precision < 3 or len(problem1) != 5
    print(f"i{' '*(order-1)}  {name1}  {' '*(precision-3)}{name2}")
    for candidate in range(dimension):
        print(
            f"{candidate+1:0{order}d}  "
            f"{portfolio1[candidate]:.{precision}f}  "
            f"{portfolio2[candidate]:.{precision}f}"
        )

    def obj_string(name: str, value: float, precision: int,
                   z: float | None = None) -> str:
        """Helper function which handles the optional z1 and z2 values"""
        obj_str = f"{name}: {value:.{precision}f}"
        return f"{obj_str} (z = {z:.{precision}f})" if z else f"{obj_str}"

    portfolio_abs_diff = np.abs(portfolio1-portfolio2)
    print(
        f"\n{obj_string(f'{name1} objective', objective1, precision, z1)}"
        f"\n{obj_string(f'{name2} objective', objective2, precision, z2)}"
        f"\nMaximum change: {max(portfolio_abs_diff):.{precision}f}"
        f"\nAverage change: {np.mean(portfolio_abs_diff):.{precision}f}"
        f"\nMinimum change: {min(portfolio_abs_diff):.{precision}f}"
    )
