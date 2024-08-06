# -*- coding: utf-8 -*-
"""Working with Pedigree Data

With pedigree data we can use Wright's Numerator Relationship Matrix to
obtain a relationship matrix and its inverse cheaply, the functions for
which are included in `pedigree`.

Documentation is available in the docstrings and online at
https://github.com/Foggalong/RobustOCS/wiki
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
    Constructs the Cholesky factor L of Wright's Numerator Relationship Matrix
    (WNRM) from a given pedigree structure.

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        The Cholesky factor L of Wright's Numerator Relationship Matrix A,
        so that A = L'L.
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
    Constructs the inverse of the D² factor from Henderson's decomposition
    (1976) of Wright's Numerator Relationship Matrix (WNRM) from a given
    pedigree structure.

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        The inverse of matrix D² in A = L'L = T'DDT, where A is Wright's
        Numerator Relationship Matrix and L is its Cholesky factor.
    """

    m: int = len(pedigree)
    L: npt.NDArray[np.floating] = makeL(pedigree)

    # don't store the full matrix, only the diagonal
    return np.array([1/L[i, i]**2 for i in range(m)], dtype=np.floating)


def make_invT(pedigree: dict[int, list[int]]) -> npt.NDArray[np.floating]:
    """
    Constructs the inverse of the T factor from Henderson's decomposition
    (1976) of Wright's Numerator Relationship Matrix (WNRM) from a given
    pedigree structure.

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        The inverse of matrix T in A = L'L = T'DDT, where A is Wright's
        Numerator Relationship Matrix and L is its Cholesky factor.
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
    Compute the inverse of Wright's Numerator Relationship Matrix (WNRM)
    using a shortcut which exploits properties of the matrix's decomposition
    as outlined by Henderson (1976).

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        The inverse of the WNRM described by the pedigree structure.
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
    Compute the inverse of Wright's Numerator Relationship Matrix (WNRM)
    using its LDDL' decomposition. NOTE: this will be less efficient than
    `make_invA`, which does this while also exploiting properties of that
    decomposition to reduce the number of computations necessary.

    Parameters
    ----------
    pedigree : dict
        Pedigree structure in `{int: [int, int]}` dictionary format, such
        as that returned from `load_ped`.

    Returns
    -------
    ndarray
        The inverse of the WNRM described by the pedigree structure.
    """

    invD2: npt.NDArray[np.floating] = make_invD2(pedigree)
    invT: npt.NDArray[np.floating] = make_invT(pedigree)

    # Since invD2 is a 1D array with shape (n,) we know invD2.reshape(-1, 1)
    # gives a column vector with shape (n,1). This means the np.multiply is
    # a Hadamard product between that vector and invT, which is equivalent
    # to diag(invD2)*invT but without forming a full matrix unnecessarily.
    return invT.transpose() @ np.multiply(invD2.reshape(-1, 1), invT)
