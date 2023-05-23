import numpy as np
from scipy.linalg import sqrtm


def get_scaling_matrix(cov_matrix: np.array) -> np.array:
    """
    Obtaining a linear normalization matrix.

    Parameters
    ----------
    cov_matrix : numpy.array
        Covariance matrix.
    """

    return np.linalg.inv(sqrtm(cov_matrix))


def get_matrix_entropy(matrix: np.array) -> float:
    """
    Calculate entropy alternation under a linear mapping.

    Parameters
    ----------
    matrix : numpy.array
        Matrix of linear mapping.
    """

    sign, logdet = np.linalg.slogdet(matrix)
    return logdet