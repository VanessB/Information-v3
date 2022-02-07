import numpy as np
from scipy.linalg import sqrtm


def get_scaling_matrix(cov_matrix):
    """
    Получение линейного преобразования, под действием которого ковариационная матрица
    переходит в единичную.

    cov_matrix - ковариационная матрица.
    """

    return np.linalg.inv(sqrtm(cov_matrix))


def get_matrix_entropy(matrix):
    """
    Вычисление изменения энтропии при линейном преобразовании, заданном матрицей.

    matrix - матрица преобразования.
    """

    sign, logdet = np.linalg.slogdet(matrix)
    return logdet