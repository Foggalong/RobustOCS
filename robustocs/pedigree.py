# -*- coding: utf-8 -*-
"""Working with Pedigree Data

TODO update this docstring.
Script which reads in a pedigree datafile and returns Wright's Numerator
Relationship Matrix for that pedigree structure.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
from math import sqrt

__all__ = ["makeA", "make_invA"]


def makeA(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Constructs Wright's Numerator Relationship Matrix (WNRM) from a given
    pedigree structure.

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        Wright's Numerator Relationship Matrix.
    """

    m: int = len(pedigree)
    # preallocate memory for A
    A: npt.NDArray[np.floating] = np.zeros((m, m), dtype=np.floating)

    # iterate over rows
    for i in range(0, m):
        # save parent indexes: pedigrees indexed from 1, Python from 0
        p: int = pedigree[i+1][0]-1
        q: int = pedigree[i+1][1]-1
        # iterate over columns sub-diagonal
        for j in range(0, i):
            # calculate sub-diagonal entries
            A[i, j] = 0.5*(A[j, p] + A[j, q])
            # populate sup-diagonal (symmetric)
            A[j, i] = A[i, j]
        # calculate diagonal entries
        A[i, i] = 1 + 0.5*A[p, q]

    return A


def makeL(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Construct the Cholesky factor L of Wright's Numerator Relationship Matrix
    from a given pedigree structure. Takes the pedigree as a dictionary input
    and returns the matrix L (in A = L'L) as output.
    # TODO update the docstring to the new standard
    """

    m: int = len(pedigree)
    L: npt.NDArray[np.floating] = np.zeros((m, m), dtype=np.floating)

    # iterate over rows
    for i in range(0, m):
        # save parent indexes: pedigrees indexed from 1, Python from 0
        p: int = pedigree[i+1][0]-1
        q: int = pedigree[i+1][1]-1

        # case where both parents are known; p < q bny *.ped convention
        if p >= 0 and q >= 0:
            for j in range(0, p+1):
                L[i, j] = 0.5*(L[p, j] + L[q, j])
            for j in range(p+1, q+1):
                L[i, j] = 0.5*L[q, j]
            # and L[i,j] = 0 for j = (q+1):i

            # compute the diagonal
            s: float = 1
            for j in range(0, p+1):
                s += 0.5*L[p, j]*L[q, j]
            for j in range(0, q+1):
                s -= L[i, j]**2
            L[i, i] = s**0.5

        # case where one parent is known; p by *.ped convention
        elif p >= 0:  # and q = 0
            for j in range(0, p+1):
                L[i, j] = 0.5*L[p, j]
            # and L[i,j] = 0 for j = (q+1):i

            # compute the diagonal
            L[i, i] = sqrt(1 - sum(L[i, j]**2 for j in range(p+1)))

        # case where neither parents known; p = q = 0)
        else:
            # L[i, j] = 0 for j in range(0, i)
            L[i, i] = 1

    return L


def make_invD2(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Construct the inverse of the D^2 factor from the Henderson (1976)
    decomposition of a WNRM. Takes the pedigree as a dictionary input
    and returns the inverse of matrix D^2 (in A = L'L = T'DDT) as output.
    # TODO update the docstring to the new standard
    """

    m: int = len(pedigree)
    L: npt.NDArray[np.floating] = makeL(pedigree)

    # don't store the full matrix, only the diagonal
    return np.array([1/L[i, i]**2 for i in range(m)], dtype=np.floating)


def make_invT(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Construct the inverse of the D factor from the Henderson (1976)
    decomposition of a WNRM. Takes the pedigree as a dictionary input
    and returns the inverse of matrix D (in A = L'L = T'DDT) as output.
    # TODO update the docstring to the new standard
    """

    m: int = len(pedigree)
    invT: npt.NDArray[np.floating] = np.zeros((m, m), dtype=np.floating)

    for i in range(0, m):
        # label parents p & q
        p: int = pedigree[i+1][0]-1
        q: int = pedigree[i+1][1]-1
        # set columns corresponding to known parents to -0.5
        if p >= 0:
            invT[i, p] = -0.5
        if q >= 0:
            invT[i, q] = -0.5
        # T^-1 has 1s on the diagonal
        invT[i, i] = 1

    return invT


def make_invA(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Compute the inverse of A using a shortcut which exploits
    of its T and D decomposition, detailed in Henderson (1976).
    Takes the pedigree as a dictionary input and returns the
    inverse as matrix output.
    # TODO update the docstring to the new standard
    """

    m: int = len(pedigree)
    invD2: npt.NDArray[np.floating] = make_invD2(pedigree)

    # convert invD2 into a dense matrix as the basis for invA
    invA = np.diag(invD2)

    for i in range(0, m):
        # label parents p & q
        p: int = pedigree[i+1][0]-1
        q: int = pedigree[i+1][1]-1

        # case where both both parents are known
        if p >= 0 and q >= 0:
            x: float = -0.5*invD2[i]
            y: float = 0.25*invD2[i]
            invA[p, i] += x
            invA[i, p] += x
            invA[q, i] += x
            invA[i, q] += x
            invA[p, p] += y
            invA[p, q] += y
            invA[q, p] += y
            invA[q, q] += y

        # case where one parent is known; p by *.ped convention
        elif p >= 0:
            x: float = -0.5*invD2[i]
            invA[p, i] += x
            invA[i, p] += x
            invA[p, p] += 0.25*invD2[i]

        # nothing to do in case where neither parent is known

    return invA


def make_invA_decomposition(
    pedigree: dict[int, list[int]]
) -> npt.NDArray[np.floating]:
    """
    Calculate the inverse of A using its T and D decomposition
    factors from Henderson (1976). Takes the pedigree as a
    dictionary input and returns the inverse as matrix output.
    # TODO update the docstring to the new standard
    """

    invD2: npt.NDArray[np.floating] = make_invD2(pedigree)
    invT: npt.NDArray[np.floating] = make_invT(pedigree)

    # computing A^-1 = (T^-1)' * (D^2)^-1 * T^-1 in full
    # 1. invD2.reshape(-1,1) converts invD2 into a 1D column array
    # 2. np.multiply computes the Hamard product between that 1D array
    #    and invT, which is equivalent to diag(invD2)*invT but without
    #    forming a full matrix unnecessarily.
    # 3. finally compute the final matrix-matrix product
    return invT.transpose() @ np.multiply(invD2.reshape(-1, 1), invT)
