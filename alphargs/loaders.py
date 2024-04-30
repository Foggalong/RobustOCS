# -*- coding: utf-8 -*-
"""Loading in Genetics Data

Genetics data could be presented to our solvers in multiple formats, these
functions define the appropriate methods for loading those in correctly.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy


def load_ped(filename: str) -> dict:  # TODO add dict typing
    """
    Function for reading *.ped files to a dictionary. Takes the file name
    as a string input and returns the pedigree structure as a dictionary.
    """
    with open(filename, "r") as file:
        # first line of *.ped lists the headers; skip
        file.readline()
        # create a list of int lists from each line (dropping optional labels)
        data = [[int(x) for x in line.split(",")[0:3]] for line in file]
    # convert this list of lists into a dictionary
    ped = {entry[0]: entry[1:3] for entry in data}

    return ped


def makeA(pedigree: dict) -> npt.NDArray[np.float64]:  # TODO add dict typing
    """
    Construct Wright's Numerator Relationship Matrix from a given pedigree
    structure. Takes the pedigree as a dictionary input and returns the
    matrix as output.
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


def load_problem(A_filename: str, E_filename: str, S_filename: str,
                 dimension: int | None = None, pedigree: bool = False
                 ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                            npt.NDArray[np.float64], int]:
    """
    Used to load genetic selection problems into NumPy. It takes three
    string inputs for filenames where Sigma, Mu, and Omega are stored,
    as well as an optional integer input for problem dimension if this
    is known. If it's know know, it's worked out based on E_filename.

    As output, it returns (A, E, S, n), where A and S are n-by-n NumPy
    arrays, E is a length n NumPy array, and n is an integer.
    """

    def load_symmetric_matrix(filename: str, dimension: int
                              ) -> npt.NDArray[np.float64]:
        """
        Since NumPy doesn't have a stock way to load matrices
        stored in coordinate format format, this adds one.
        """

        matrix = np.zeros([dimension, dimension], dtype=float)

        with open(filename, 'r') as file:
            for line in file:
                i, j, entry = line.split(" ")
                # data files indexed from 1, not 0
                matrix[int(i)-1, int(j)-1] = entry
                matrix[int(j)-1, int(i)-1] = entry

        return matrix

    E = np.loadtxt(E_filename, dtype=float)
    # if dimension not specified, use `E` which doesn't need preallocation
    if not dimension:
        assert isinstance(E.size, int)  # catches E being empty
        dimension = E.size

    # S is stored by coordinates so need special loader
    S = load_symmetric_matrix(S_filename, dimension)
    # A can be stored as a pedigree or by coordinates
    if pedigree:
        A = makeA(load_ped(A_filename))
    else:
        A = load_symmetric_matrix(A_filename, dimension)

    return A, E, S, dimension
