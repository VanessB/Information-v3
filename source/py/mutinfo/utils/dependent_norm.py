import numpy as np

def get_norm_corr_coef_from_MI(mutual_information):
    """
    Получение коэффициента корреляции для двух нормальных случайных величин
    по их взаимной информации.

    mutual_information - взаимноая информация (в диапазоне [0.0; +inf))
    """

    assert 0.0 <= mutual_information

    return np.sqrt(1 - np.exp(- 2.0 * mutual_information))
