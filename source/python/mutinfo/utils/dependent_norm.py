import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from scipy.linalg import block_diag


def norm_corr_from_MI(mutual_information: float) -> float:
    """
    Получение коэффициента корреляции для двух нормальных случайных величин
    по их взаимной информации.

    Параметры
    ---------
    mutual_information : float
        Взаимноая информация (в диапазоне [0.0; +inf)).
    """

    if mutual_information < 0.0:
        raise ValueError("Mutual information must be non-negative")

    return np.sqrt(1 - np.exp(- 2.0 * mutual_information))


def multivariate_normal_from_MI(X_dimension: int, Y_dimension: int, mutual_information: float,
                                X_rotate: bool=True, Y_rotate: bool=True) -> multivariate_normal:
    """
    Получение нормального случайного вектора размерности X_dimension + Y_dimension
    с заданной взаимной информацией между первыми X_dimension и последними Y_dimension
    компонентами.

    Параметры
    ---------
    X_dimension : int
        Размерность первого случайного вектора.
    Y_dimension : int
        Размерность второго случайного вектора.
    mutual_information : float
        Взаимноая информация (в диапазоне [0.0; +inf)).
    X_rotate : bool
        Применять ли случайное вращение к первому вектору.
    Y_rotate : bool
        Применять ли случайное вращение ко второму вектору.
    """

    if X_dimension < 1 or Y_dimension < 1:
        raise ValueError("X and Y dimensions must be at least 1.")

    # Разбивка зависимости равномерно по компонентам.
    min_dim = min(X_dimension, Y_dimension)
    sum_dim = X_dimension + Y_dimension
    MI_per_dim = mutual_information / min_dim

    # Построение базовой матрицы ковариации.
    corr_coef = norm_corr_from_MI(MI_per_dim)
    cov_matrix = np.identity(sum_dim)
    for index in range(min_dim):
        cov_matrix[index][index + X_dimension] = corr_coef
        cov_matrix[index + X_dimension][index] = corr_coef

    # Случайное вращение внутри каждой группы.
    if X_dimension > 1 and X_rotate:
        Q_X = ortho_group.rvs(X_dimension)
    else:
        Q_X = np.identity(X_dimension)

    if Y_dimension > 1 and Y_rotate:
        Q_Y = ortho_group.rvs(Y_dimension)
    else:
        Q_Y = np.identity(Y_dimension)

    Q = block_diag(Q_X, Q_Y)

    return multivariate_normal(cov = Q @ cov_matrix @ Q.T)