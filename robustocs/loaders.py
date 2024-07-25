# -*- coding: utf-8 -*-
"""Loading in Genetics Data

Genetics data could be presented to our solvers in multiple formats, these
functions define the appropriate methods for loading those in correctly.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
from scipy import sparse    # used for sparse matrix format

# controls what's imported on `from robustocs.loaders import *`
__all__ = ["load_ped", "makeA", "load_problem"]


# DATA LOADERS
# The structures we're working with are either stored in symmetric sparse
# coordinate format (i.e. COO format, but only storing those above/below the
# diagonal) or as pedigree files. Neither Numpy or SciPy have functions for
# loading these data formats, so these are written below.

def count_sparse_nnz(filename: str) -> int:
    """
    Return the number of non-zero entries for a matrix stored in a file
    in symmetric sparse coordinate (COO) format.

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
                          ) -> npt.NDArray[np.floating]:
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

    matrix = np.zeros([dimension, dimension], dtype=np.floating)

    with open(filename, 'r') as file:
        for line in file:
            i, j, entry = line.split(" ")
            # data files indexed from 1, not 0
            matrix[int(i)-1, int(j)-1] = entry
            matrix[int(j)-1, int(i)-1] = entry

    return matrix


def load_sparse_symmetric_matrix(filename: str, dimension: int, nnz: int,
                                 format: str = 'csr') -> sparse.spmatrix:
    """
    Since neither NumPy or SciPy have a stock way to load symmetric matrices
    stored in symmetric coordinate format into SciPy's formats, this adds one.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.
    dimension : int
        Number of rows (and columns) of the matrix stored in the file.
    nnz : int
        Number of non-zero entries in the matrix stored in the file. Note that
        due to symmetry, this may be larger than the number of lines in file.
    format : str
        Format to use for the sparse matrix. Options available are 'coo' (for
        sparse coordinate format) or 'csr' (compressed sparse row format).
        Default value is `csr`.

    Returns
    -------
    spmatrix
        The matrix represented by the file.
    """

    # preallocate storage arrays
    rows: npt.NDArray[np.integer] = np.zeros(nnz, dtype=np.integer)
    cols: npt.NDArray[np.integer] = np.zeros(nnz, dtype=np.integer)
    vals: npt.NDArray[np.floating] = np.zeros(nnz, dtype=np.floating)

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

    if format == 'coo':
        return sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=(dimension, dimension),
            dtype=np.floating
        )
    elif format == 'csr':
        return sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(dimension, dimension),
            dtype=np.floating
        )
    else:
        raise ValueError("Format must be 'coo' or 'csr'.")


def load_sexes(filename: str, dimension: int) -> tuple[
    npt.NDArray[np.unsignedinteger],
    npt.NDArray[np.unsignedinteger],
    npt.NDArray[np.str]
]:
    """
    Function for loading cohort sex data from file. Uniquely among the
    file formats we're working with, the format for sex data also includes
    a label for each candidate in the cohort, so also returns those.

    Parameters
    ----------
    filename : str
        Filename, including extension. Space-separated data file.
    dimension : int
        Number of rows of data stored in the file

    Returns
    -------
    ndarray
        Array of indices for sires in the cohort.
    ndarray
        Array of indices for dams in the cohort.
    ndarray
        Array of labels for each candidate in the cohort.
    """

    # preallocate output vectors
    sires = np.zeros((dimension,), dtype=np.unsignedinteger)
    dams = np.zeros((dimension,), dtype=np.unsignedinteger)
    names = np.zeros((dimension,), dtype=np.str)

    # index trackers
    sire_count: int = 0
    dam_count: int = 0

    with open(filename, 'r') as file:
        for line in file:
            name, sex = line.strip().split(" ")

            if sex == "M":
                sires[sire_count] = sire_count + dam_count
                sire_count += 1
            elif sex == "F":
                dams[dam_count] = sire_count + dam_count
                dam_count += 1
            else:
                raise ValueError(f"invalid sex input {sex}")

            names[sire_count + dam_count - 1] = name

    # cull the entries of the index vectors which weren't used
    return sires[:sire_count], dams[:dam_count], names


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


# PROBLEM LOADERS
# Combine all of the above functions to give an easy interface for loading
# problems into Python. At the moment this is through a single function,
# though may need to be split if more features are added.

def load_problem(
    sigma_filename: str,
    mu_filename: str,
    omega_filename: str | None = None,
    sex_filename: str | None = None,
    nnz_sigma: int | None = None,
    nnz_omega: int | None = None,
    dimension: int | None = None,
    pedigree: bool = False,
    issparse: bool = False
) -> tuple[
    npt.NDArray[np.floating] | sparse.spmatrix,
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | sparse.spmatrix | None,
    int,
    npt.NDArray[np.unsignedinteger] | None,
    npt.NDArray[np.unsignedinteger] | None,
    npt.NDArray[np.str] | None
]:
    """
    Load a robust genetic selection problem into Python.

    Parameters
    ----------
    sigma_filename : str
        Filename for a file which encodes sigma, whether that's in sparse
        coordinate format or pedigree format.
    mu_filename : str
        Filename for a file which encodes mu or mubar.
    omega_filename : str, optional
        Filename for a file which encodes omega, which will be in sparse
        coordinate format. Default value is `None`.
    sex_filename : str, optional
        Filename for a file which encodes the sex data for the cohort. Also
        includes a label for each candidate. Default value is `None`.
    nnz_sigma : int, optional
        Number of non-zeros in sigma. If not provided and `sparse` is
        True, value computed using `count_sparse_nnz`. Default is `None`.
    nnz_omega : int, optional
        Number of non-zeros in omega. If not provided and `sparse` is
        True, value computed using `count_sparse_nnz`. Default is `None`.
    dimension : int, optional
        The size of the problem, which can be specified to aid preallocation
        or worked out explicitly from the mu / mubar produced. Default value
        is `None`, i.e. the value is derived from mu.
    pedigree : bool, optional
        Signifies whether sigma is stored as a pedigree structure (`True`)
        or in sparse coordinate format (`False`). Default value is `False`.
    issparse : bool, optional
        Signifies whether sigma and omega should be returned in compressed
        sparse row format. Default value is `False`.

    Returns
    -------
    ndarray or spmatrix
        Covariance matrix of candidates in the cohort (sigma).
    ndarray
        Vector of expected values of the expected breeding values of
        candidates in the cohort (mu).
    ndarray, spmatrix, or None
        Covariance matrix of expected breeding values of candidates in the
        cohort (omega). If a filename was not provided, value is `None`.
    int
        Dimension of the problem (n).
    ndarray or None
        Array of indices of sires in the cohort (S). If a filename for sex data
        was not provided, value is `None`.
    ndarray or None
        Array of indices of dams in the cohort (D). If a filename for sex data
        was not provided, value is `None`.
    ndarray or None
        Array of names given to the original candidates in the cohort. If a
        filename for sex data was not provided (which also includes name data),
        then the value is `None`.
    """

    mu: npt.NDArray[np.floating] = np.loadtxt(mu_filename, dtype=np.floating)
    # if dimension not specified, use `mu` which doesn't need preallocation
    if not dimension:
        assert isinstance(mu.size, int)  # catches mu being empty
        dimension = mu.size

    # filename for sex data is optional, so skip if not provided
    if sex_filename is not None:
        sires, dams, names = load_sexes(sex_filename, dimension)
    else:
        sires = dams = names = None

    # filename for omega is optional, so skip if not provided
    if omega_filename is not None:
        # can load omega from file to SciPy's CSR or NumPy's dense format
        if issparse:
            # find omega's number of non zeros if it wasn't given
            if not nnz_omega:
                nnz_omega = count_sparse_nnz(omega_filename)
            omega: sparse.spmatrix = load_sparse_symmetric_matrix(
                omega_filename, dimension, nnz_omega
            )
        else:
            # if nnz_omega was defined, it's ignored as a parameter
            omega: npt.NDArray[np.floating] = load_symmetric_matrix(
                omega_filename, dimension
            )

    # sigma can be stored as a pedigree or by coordinates and can be loaded to
    # SciPy's CSR or Numpy's dense format. Hence have four branches below.
    if pedigree:
        sigma: npt.NDArray[np.floating] = makeA(load_ped(sigma_filename))
        # HACK this loads the full matrix, then converts it down to sparse
        if issparse:
            sigma: sparse.spmatrix = sparse.coo_matrix(sigma)
    else:
        if issparse:
            # find sigma's number of non zeros if it wasn't given
            if not nnz_sigma:
                nnz_sigma = count_sparse_nnz(sigma_filename)
            sigma: sparse.spmatrix = load_sparse_symmetric_matrix(
                sigma_filename, dimension, nnz_sigma
            )
        else:
            # if nnz_sigma was defined here, it's ignored as a parameter
            sigma: npt.NDArray[np.floating] = load_symmetric_matrix(
                sigma_filename, dimension
            )

    return sigma, mu, omega, dimension, sires, dams, names
