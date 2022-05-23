import csv
import math
import numpy as np
import os
import shutil


from .functional import KDEFunctional, KLFunctional
from ..utils.matrices import get_scaling_matrix, get_matrix_entropy


class EntropyEstimator:
    """
    Класс-эстиматор дифференциальной энтропии, использующий ядерную оценку плотности
    и метод убрать-один-элемент.
    """

    def __init__(self, rescale=True, n_jobs=1, estimation_rtol=0.0):
        self.n_jobs = n_jobs
        self.estimation_rtol = estimation_rtol
        self.best_estimator_ = None

        self.rescale = rescale
        self.scaling_matrix_ = None
        self.scaling_delta_entropy_ = 0.0


    def fit(self, data, fit_scaling_matrix=True,
            fit_bandwidth=True, verbose=0):
        """
        Подбор параметров ядерной оценки плотности по данным.

        data - выборка из исследуемой случайной величины.
        fit_scaling_matrix - вычислить матрицу масштабирования.
        fit_bandwidth - подобрать оптимальную ширину окна.
        verbose - подробность вывода.
        """

        # Матрица ковариации (требуется для нормировки).
        # Учитывается, что в случае одномерных данных np.cov возвращает число.
        cov_matrix = np.cov(data, rowvar=False)
        if data.shape[1] == 1:
            cov_matrix = np.array([[cov_matrix]])

        # Нормировка данных.
        if self.rescale:
            if fit_scaling_matrix:
                self.scaling_matrix_ = get_scaling_matrix(cov_matrix)
                self.scaling_delta_entropy_ = -0.5 * get_matrix_entropy(cov_matrix)

            _data = data @ self.scaling_matrix_
            min_std = 1.0
            max_std = 1.0
        else:
            _data = data
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            min_std = np.sqrt(eigenvalues[0])
            max_std = np.sqrt(eigenvalues[-1])
            
        # Класс для оценки функционала.
        #self.best_estimator_ = KDEFunctional(n_jobs=self.n_jobs)
        self.best_estimator_ = KLFunctional(n_jobs=self.n_jobs)
        self.best_estimator_.fit(_data)

        # Подбор оптимальной ширины окна.
        #if fit_bandwidth:
        #    self.best_estimator_.set_optimal_bandwidth(verbose=verbose)


    def predict(self, data, first_n_elements=None, n_parts=10, batch_size=32,
            save_intermediate=False, recover_saved=False, verbose=0):
        """
        Параллельное вычисление оценки энтропии методом убрать-один-элемент.

        data              - выборка из исследуемой случайной величины.
        first_n_elements  - оценка интеграла только по первым first_n_elements точкам.
                            (если параметр равен None, оценка производится по всем точкам)
        n_parts           - число частей, сохраняемых во временные файлы на случай
                            восстановления после сбоя.
        batch_size        - размер батча при распараллеливании.
        save_intermediate - сохранение промежуточных данных во временные файлы.
        recover_saved     - произвести восстановление из временных файлов.
        verbose           - подробность вывода.
        """

        # Изменение точности оценщика.
        #self.best_estimator_.set_params(rtol = self.estimation_rtol)

        # Нормировка данных.
        _data = data @ self.scaling_matrix_ if self.rescale else data

        mean, std = self.best_estimator_.integrate(np.log)

        return -mean - self.scaling_delta_entropy_, std