import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from scipy.linalg import block_diag


def norm_corr_from_MI(mutual_information):
    """
    Получение коэффициента корреляции для двух нормальных случайных величин
    по их взаимной информации.

    mutual_information - взаимноая информация (в диапазоне [0.0; +inf))
    """

    assert 0.0 <= mutual_information

    return np.sqrt(1 - np.exp(- 2.0 * mutual_information))


def multivariate_normal_from_MI(X_dimension, Y_dimension, mutual_information,
                                X_rotate = True, Y_rotate = True):
    """
    Получение нормального случайного вектора размерности dimension_X + dimension_Y
    с заданной взаимной информацией между первыми dimension_X и последними dimension_Y
    компонентами.

    mutual_information - взаимноая информация (в диапазоне [0.0; +inf))
    """

    assert X_dimension >= 1
    assert Y_dimension >= 1

    # Разбивка зависимости равномерно по компонентам.
    min_dim = min(X_dimension, Y_dimension)
    sum_dim = X_dimension + Y_dimension
    MI_per_dim = mutual_information / min_dim

    # Построение базовой матрицы ковариации.
    corr_coef = norm_corr_from_MI(MI_per_dim)
    cov_matrix = np.identity(sum_dim)
    for component in range(min_dim):
        cov_matrix[component][component + X_dimension] = corr_coef
        cov_matrix[component + X_dimension][component] = corr_coef

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