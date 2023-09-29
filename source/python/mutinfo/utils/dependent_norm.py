import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from scipy.linalg import block_diag


def norm_corr_from_MI(mutual_information: float) -> float:
    """
    Correlation coefficient for two normal random variables,
    given the value of mutual information between them.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).
    """

    if mutual_information < 0.0:
        raise ValueError("Mutual information must be non-negative")

    return np.sqrt(1 - np.exp(- 2.0 * mutual_information))


def multivariate_normal_from_MI(X_dimension: int, Y_dimension: int,
                                mutual_information: float,
                                mix_components: bool=True) -> multivariate_normal:
    """
    Obtaining a normal random vector of `X_dimension + Y_dimension` dimension
    with the given mutual information between the first X_dimension and the
    last Y_dimension components.

    Parameters
    ----------
    X_dimension : int
        The dimensionality of the first random vector.
    Y_dimension : int
        The dimensionality of the second random vector.
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).
    mix_components : bool
        Apply random unitary mapping to both of the vector.
    """

    if X_dimension < 1 or Y_dimension < 1:
        raise ValueError("X and Y dimensions must be at least 1")

    # Split mutual information between components.
    min_dim = min(X_dimension, Y_dimension)
    sum_dim = X_dimension + Y_dimension
    MI_per_dim = mutual_information / min_dim

    # Build the covariance matrix.
    corr_coef = norm_corr_from_MI(MI_per_dim)
    cov_matrix = np.identity(sum_dim)
    for index in range(min_dim):
        cov_matrix[index][index + X_dimension] = corr_coef
        cov_matrix[index + X_dimension][index] = corr_coef

    # Random rotation of the corresponding vectors.
    if X_dimension > 1 and mix_components:
        Q_X = ortho_group.rvs(X_dimension)
    else:
        Q_X = np.identity(X_dimension)

    if Y_dimension > 1 and mix_components:
        Q_Y = ortho_group.rvs(Y_dimension)
    else:
        Q_Y = np.identity(Y_dimension)

    Q = block_diag(Q_X, Q_Y)

    return multivariate_normal(cov = Q @ cov_matrix @ Q.T)