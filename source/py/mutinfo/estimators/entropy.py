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

    def __init__(self, rescale=True, method='KDE', functional_params=None):
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
        functional_params : dict
            Параметры функционала.
        """

        self.rescale = rescale
        self.scaling_matrix_ = None
        self.scaling_delta_entropy_ = 0.0
        
        if method == 'KDE':
            self.functional_ = KDEFunctional(**functional_params)
        if method == 'KL':
            self.functional_ = KLFunctional(**functional_params)


    def fit(self, data, fit_scaling_matrix=True,
            verbose=0, **kwargs):
        """
        Предобработка данных и подбор параметров оценщика функционала

        Параметры
        ---------
        data : array
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
                self.scaling_matrix_ = get_scaling_matrix(cov_matrix)
                self.scaling_delta_entropy_ = -0.5 * get_matrix_entropy(cov_matrix)

            data = data @ self.scaling_matrix_
            
        # Оценщик функционала.
        self.functional_.fit(data, **kwargs, verbose=verbose)

            
    def estimate(self, data, verbose=0):
        """
        Оценка энтропии.
        
        Параметры
        ---------
        verbose : int
            Подробность вывода.
        """

        # Сама оценка производится оценщиком функционала.
        mean, std = self.functional_.integrate(np.log, verbose=verbose)

        return -mean - self.scaling_delta_entropy_, std