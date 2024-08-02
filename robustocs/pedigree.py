# -*- coding: utf-8 -*-
"""Working with Pedigree Data

TODO update this docstring.
Script which reads in a pedigree datafile and returns Wright's Numerator
Relationship Matrix for that pedigree structure.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy


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

    m = len(pedigree)
    # preallocate memory for A
    A = np.zeros((m, m), dtype=np.floating)

    # iterate over rows
    for i in range(0, m):
        # save parent indexes: pedigrees indexed from 1, Python from 0
        p = pedigree[i+1][0]-1
        q = pedigree[i+1][1]-1
        # iterate over columns sub-diagonal
        for j in range(0, i):
            # calculate sub-diagonal entries
            A[i, j] = 0.5*(A[j, p] + A[j, q])
            # populate sup-diagonal (symmetric)
            A[j, i] = A[i, j]
        # calculate diagonal entries
        A[i, i] = 1 + 0.5*A[p, q]

    return A


def makeL(pedigree):
    """
    Construct the Cholesky factor L of Wright's Numerator Relationship Matrix
    from a given pedigree structure. Takes the pedigree as a dictionary input
    and returns the matrix L (in A = L'L) as output.
    """
    m = len(pedigree)
    L = np.zeros((m, m))

    # iterate over rows
    for i in range(0, m):
        # save parent indexes: pedigrees indexed from 1, Python from 0
        p = pedigree[i+1][0]-1
        q = pedigree[i+1][1]-1

        # case where both parents are known; p < q bny *.ped convention
        if p >= 0 and q >= 0:
            for j in range(0, p+1):
                L[i, j] = 0.5*(L[p, j] + L[q, j])
            for j in range(p+1, q+1):
                L[i, j] = 0.5*L[q, j]
            # and L[i,j] = 0 for j = (q+1):i

            # compute the diagonal
            s = 1
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
            s = 1
            for j in range(0, p+1):
                s -= L[i, j]**2
            L[i, i] = s**0.5

        else:
            for j in range(0, i):
                L[i, j] = 0
            L[i, i] = 1

    return(L)


def make_invD2(pedigree):
    """
    Construct the inverse of the D^2 factor from the Henderson (1976)
    decomposition of a WNRM. Takes the pedigree as a dictionary input
    and returns the inverse of matrix D^2 (in A = L'L = T'DDT) as output.
    """
    m = len(pedigree)
    L = makeL(pedigree)
    invD2 = np.zeros((m, m))  # TODO find a way to store diagonal matrices

    # iterate over rows
    for i in range(0, m):
        invD2[i, i] = 1/(L[i, i]**2)

    return(invD2)


def make_invT(pedigree):
    """
    Construct the inverse of the D factor from the Henderson (1976)
    decomposition of a WNRM. Takes the pedigree as a dictionary input
    and returns the inverse of matrix D (in A = L'L = T'DDT) as output.
    """
    m = len(pedigree)
    invT = np.zeros((m, m))

    for i in range(0, m):
        # label parents p & q
        p = pedigree[i+1][0]-1
        q = pedigree[i+1][1]-1
        # set columns corresponding to known parents to -0.5
        if p >= 0:
            invT[i, p] = -0.5
        if q >= 0:
            invT[i, q] = -0.5
        # T^-1 has 1s on the diagonal
        invT[i, i] = 1

    return(invT)


def make_invA(pedigree):
    """
    Compute the inverse of A using a shortcut which exploits
    of its T and D decomposition, detailed in Henderson (1976).
    Takes the pedigree as a dictionary input and returns the
    inverse as matrix output.
    """
    m = len(pedigree)
    B = make_invD2(ped)
    invA = B

    for i in range(0, m):
        # label parents p & q
        p = pedigree[i+1][0]-1
        q = pedigree[i+1][1]-1

        # case where both both parents are known
        if p >= 0 and q >= 0:
            x = -0.5*B[i, i]
            y = 0.25*B[i, i]
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
            x = -0.5*B[i, i]
            invA[p, i] += x
            invA[i, p] += x
            invA[p, p] += 0.25*B[i, i]

    return(invA)


def make_invA_decomposition(pedigree):
    """
    Calculate the inverse of A using its T and D decomposition
    factors from Henderson (1976). Takes the pedigree as a
    dictionary input and returns the inverse as matrix output.
    """
    invD2 = make_invD2(ped)
    invT = make_invT(ped)

    # computing A^-1 = (T^-1)' * (D^2)^-1 * T^-1 in full
    invA = np.matmul(invT.transpose(), np.matmul(invD2, invT))
    return(invA)
