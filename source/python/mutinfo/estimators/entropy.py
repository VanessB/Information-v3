"""
Реализация оценщика энтропии.

При оценке энтропии требуется отнормировать данные, подобрать параметры для
численного интегрирования и проинтегрировать логарифм плотности вероятности.
"""

import math
import numpy as np

from .functional import KDEFunctional, KLFunctional
from ..utils.matrices import get_scaling_matrix, get_matrix_entropy


class EntropyEstimator:
    """
    Класс-оценщик дифференциальной энтропии.
    """

    def __init__(self, rescale: bool=True, method: str='KDE',
                 functional_params: dict=None):
        """
        Инициализация экземпляра класса.
        
        Параметры
        ---------
        rescale : bool
            Требуется ли приводить данные к отмасштабированному нескоррелированному виду.
        method : str
            Способ оценки значения функционала
              'KDE' - ядерная оценка плотности
              'KL'  - Козаченко-Леоненко
        _functional_params : dict
            Параметры функционала.
        """

        self.rescale = rescale
        self._scaling_matrix = None
        self._scaling_delta_entropy = 0.0
        
        if method == 'KDE':
            self._functional = KDEFunctional(**functional_params)
        elif method == 'KL':
            self._functional = KLFunctional(**functional_params)
        else:
            raise NotImplementedError(f"Method {method} is not implemented")


    def fit(self, data, fit_scaling_matrix: bool=True,
            verbose: int=0, **kwargs):
        """
        Предобработка данных и подбор параметров оценщика функционала

        Параметры
        ---------
        data : array_like
            Выборка из исследуемой случайной величины.
        fit_scaling_matrix : bool
            Вычислять ли матрицу масштабирования.
        fit_bandwidth : bool
            Подобирать ли оптимальную ширину окна (только для KDE-функционала).
        verbose : int
            Подробность вывода.
        """

        # Нормировка данных.
        if self.rescale:
            if fit_scaling_matrix:
                # Матрица ковариации (требуется для нормировки).
                # Учитывается, что в случае одномерных данных np.cov возвращает число.
                cov_matrix = np.cov(data, rowvar=False)
                if data.shape[1] == 1:
                    cov_matrix = np.array([[cov_matrix]])
                
                # Получение масштабирующей матрицы по матрице ковариации.
                self._scaling_matrix = get_scaling_matrix(cov_matrix)
                self._scaling_delta_entropy = -0.5 * get_matrix_entropy(cov_matrix)

            data = data @ self._scaling_matrix
            
        # Оценщик функционала.
        self._functional.fit(data, **kwargs, verbose=verbose)

            
    def estimate(self, data, verbose: int=0) -> (float, float):
        """
        Оценка энтропии.
        
        Параметры
        ---------
        verbose : int
            Подробность вывода.
        """

        # Сама оценка производится оценщиком функционала.
        mean, std = self._functional.integrate(np.log, verbose=verbose)

        return -mean - self._scaling_delta_entropy, std