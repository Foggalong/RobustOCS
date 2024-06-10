# -*- coding: utf-8 -*-
"""Loading in Genetics Data

Genetics data could be presented to our solvers in multiple formats, these
functions define the appropriate methods for loading those in correctly.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
from scipy import sparse    # used for sparse matrix format

# controls what's imported on `from alphargs.loaders import *`
__all__ = ["load_ped", "makeA", "load_problem"]


# DATA LOADERS
# The structures we're working with are either stored in symmetric sparse
# coordinate format (i.e. COO format, but only storing those above/below the
# diagonal) or as pedigree files. Neither Numpy or SciPy have functions for
# loading these data formats, so these are written below.

def count_sparse_nnz(filename: str) -> int:
    """
    Return the number of non-zero entries for a matrix stored in symmetric
    sparse coordinate format.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.

    Returns
    -------
    int
        The number of non-zeros in the matrix represented by the file.
    """

    count: int = 0

    with open(filename, 'r') as file:
        for line in file:
            # don't care about value, so can be dropped
            i, j = line.split(" ")[:2]
            # off-diagonal entries stored once, but count twice
            count += 1 if i == j else 2

    return count


def load_symmetric_matrix(filename: str, dimension: int
                          ) -> npt.NDArray[np.float64]:
    """
    Since NumPy doesn't have a stock way to load symmetric matrices stored in
    symmetric coordinate format, this adds one.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.
    dimension : int
        Number of rows (and columns) of the matrix stored in the file

    Returns
    -------
    ndarray
        The matrix represented by the file.
    """

    matrix = np.zeros([dimension, dimension], dtype=float)

    with open(filename, 'r') as file:
        for line in file:
            i, j, entry = line.split(" ")
            # data files indexed from 1, not 0
            matrix[int(i)-1, int(j)-1] = entry
            matrix[int(j)-1, int(i)-1] = entry

    return matrix


def load_symmetric_matrix_coo(filename: str, dimension: int, nnz: int
                              ) -> sparse.spmatrix:
    """
    Since neither NumPy or SciPy have a stock way to load symmetric matrices
    into sparse coordinate format, this adds one.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.
    dimension : int
        Number of rows (and columns) of the matrix stored in the file.
    nnz : int
        Number of non-zero entries in the matrix stored in the file. Note that
        due to symmetry, this may be larger than the number of lines in file.

    Returns
    -------
    ndarray
        The matrix represented by the file.
    """

    # preallocate storage arrays
    rows: npt.NDArray[np.int8] = np.zeros(nnz)
    cols: npt.NDArray[np.int8] = np.zeros(nnz)
    vals: npt.NDArray[np.float64] = np.zeros(nnz)

    with open(filename, 'r') as file:
        index: int = 0
        for line in file:
            i, j, entry = line.split(" ")
            # data files indexed from 1, not 0
            i = int(i) - 1
            j = int(j) - 1

            rows[index] = i
            cols[index] = j
            vals[index] = entry

            # done if on diagonal, otherwise need same on the sub-diagonal
            if i != j:
                index += 1
                rows[index] = j
                cols[index] = i
                vals[index] = entry

            index += 1

    return sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(dimension, dimension),
        dtype=np.float64
    )


def load_symmetric_matrix_csr(filename: str, dimension: int, nnz: int
                              ) -> sparse.spmatrix:
    """
    Loads a symmetric matrix into compressed sparse row format. It does this
    by first loading into sparse coordinate format and then converting with
    Scipy. TODO: add a more efficient way which loads into CSR directly.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.
    dimension : int
        Number of rows (and columns) of the matrix stored in the file.
    nnz : int
        Number of non-zero entries in the matrix stored in the file. Note that
        due to symmetry, this may be larger than the number of lines in file.

    Returns
    -------
    ndarray
        The matrix represented by the file.
    """

    matrix = load_symmetric_matrix_coo(filename, dimension, nnz)
    return sparse.csr_matrix(matrix)


def load_ped(filename: str) -> dict[int, list[int]]:
    """
    Function for reading pedigree files into a Python dictionary.

    Parameters
    ----------
    filename : str
        Filename, including extension. Does not have to be `.ped` specifically.

    Returns
    -------
    dict
        Represents the pedigree structure. Integer keys give the index of
        each candidate and (integer, integer) values give the indices of that
        candidates parents. An index of zero signifies unknown parentage.
    """

    with open(filename, "r") as file:
        # first line of *.ped lists the headers; skip
        file.readline()
        # create a list of int lists from each line (dropping optional labels)
        data = [[int(x) for x in line.split(",")[0:3]] for line in file]
    # convert this list of lists into a dictionary
    ped = {entry[0]: entry[1:3] for entry in data}

    return ped


# MATRIX GENERATORS
# Utility functions for generating matrices from pedigree data.

def makeA(pedigree: dict[int, list[int]]) -> npt.NDArray[np.float64]:
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
    A = np.zeros((m, m), dtype=float)

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


# PROBLEM LOADERS
# Combine all of the above functions to give an easy interface for loading
# problems into Python. At the moment this is through a single function,
# though may need to be split if more features are added.

def load_problem(A_filename: str, E_filename: str, S_filename: str,
                 nnzA: int | None = None, nnzS: int | None = None,
                 dimension: int | None = None, pedigree: bool = False,
                 issparse: bool = False
                 ) -> tuple[npt.NDArray[np.float64] | sparse.spmatrix,
                            npt.NDArray[np.float64],
                            npt.NDArray[np.float64] | sparse.spmatrix,
                            int]:
    """
    Load a robust genetic selection problem into Python.

    Parameters
    ----------
    A_filename : str
        Filename for a file which encodes `A` (which is Sigma) whether in
        sparse coordinate or pedigree format.
    E_filename : str
        Filename for a file which encodes `E` (which is Mu-bar).
    S_filename : str
        Filename for a file which encodes `S` (which is Omega) which will be
        in sparse coordinate format.
    nnzA : int, optional
        Number of non-zeros in the matrix A. If not provided and `sparse` is
        True, value computed using `count_sparse_nnz`. Default is `None`.
    nnzS : int, optional
        Number of non-zeros in the matrix S. If not provided and `sparse` is
        True, value computed using `count_sparse_nnz`. Default is `None`.
    dimension : int or None, optional
        The size of the problem, which can be specified to aid preallocation
        or worked out explicitly from the `E` / mu produced. Default value
        is `None`, i.e. the value is derived from `E`.
    pedigree : bool, optional
        Signifies whether `A` is stored as a pedigree structure (`True`)
        or in sparse coordinate format (`False`). Default value is `False`.
    issparse : bool, optional
        Signifies whether `A` and `S` should be returned in compressed sparse
        row format. Default value is `False`.

    Returns
    -------
    ndarray or spmatrix
        Covariance matrix of candidates in the cohort.
    ndarray
        Vector of expected values of the expected breeding values of
        candidates in the cohort.
    ndarray or spmatrix
        Covariance matrix of expected breeding values of candidates in the
        cohort.
    int
        Dimension of the problem.
    """

    E = np.loadtxt(E_filename, dtype=float)
    # if dimension not specified, use `E` which doesn't need preallocation
    if not dimension:
        assert isinstance(E.size, int)  # catches E being empty
        dimension = E.size

    # can load S from coordinates to SciPy's CSR or NumPy's dense format
    if issparse:
        if not nnzS:
            nnzS = count_sparse_nnz(S_filename)
        S = load_symmetric_matrix_csr(S_filename, dimension, nnzS)
    else:
        # if nnzS was defined here, it's ignored as a parameter
        S = load_symmetric_matrix(S_filename, dimension)

    # A can be stored as a pedigree or by coordinates and can be loaded to
    # SciPy's CSR or Numpy's dense format. Hence have four branches below.
    if pedigree:
        A = makeA(load_ped(A_filename))
        # HACK this loads the full matrix, then converts it down to sparse
        if issparse:
            A = sparse.coo_matrix(A)
    else:
        if issparse:
            if not nnzA:
                nnzA = count_sparse_nnz(A_filename)
            A = load_symmetric_matrix_csr(A_filename, dimension, nnzA)
        else:
            # if nnzA was defined here, it's ignored as a parameter
            A = load_symmetric_matrix(A_filename, dimension)

    return A, E, S, dimension
