import csv
import math
import numpy as np
import os
import shutil

from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from ..utils.matrices import get_scaling_matrix, get_matrix_entropy


def _find_best_bandwidth(data, min_bw, max_bw,
        KernelDensity_args = {"kernel": "gaussian", "rtol": 1e-3},
        GridSearchCV_args  = {"n_jobs": 1, "cv": 5},
        n_points=7, rtol=1e-2, verbose=0):

    """
    Рекурсивный поиск лучшей ширины ядра для kernel density estimation.
    """

    while True:
        grid = np.logspace(np.log10(min_bw), np.log10(max_bw), n_points)
        if verbose >= 1:
            print("Поиск по сетке: ", grid)
        params = {'bandwidth': grid}

        grid_search = GridSearchCV(KernelDensity(**KernelDensity_args), params, **GridSearchCV_args, verbose=verbose)
        grid_search.fit(data)

        if grid_search.best_index_ == 0:
            min_bw *= min_bw / max_bw
            max_bw  = grid[1]
        elif grid_search.best_index_ == n_points - 1:
            max_bw *= max_bw / min_bw
            min_bw  = grid[-2]
        else:
            min_bw = grid[grid_search.best_index_ - 1]
            max_bw = grid[grid_search.best_index_ + 1]

            if max_bw - min_bw < rtol * grid[grid_search.best_index_]:
                return grid_search


class EntropyEstimator:
    """
    Класс-эстиматор дифференциальной энтропии, использующий ядерную оценку плотности
    и метод убрать-один-элемент.
    """

    def __init__(self, rescale=True,
            KernelDensity_args={"kernel": "gaussian", "rtol": 1e-3},
            GridSearchCV_args={"cv": 5},
            n_jobs=1, estimation_rtol=0.0):

        self.KernelDensity_args = KernelDensity_args
        self.GridSearchCV_args  = GridSearchCV_args

        self.n_jobs = n_jobs
        self.GridSearchCV_args["n_jobs"] = self.n_jobs
        self.estimation_rtol = estimation_rtol

        self.search_results_ = None
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

        # Подбор оптимальной ширины окна.
        # Начальное приближение - Silverman's rule-of-thumb.
        bw_factor = np.power(_data.shape[0], 0.2 / _data.shape[1])
        if fit_bandwidth:
            min_bw =  0.5 * min_std / bw_factor
            max_bw = 1.06 * max_std / bw_factor

            self.search_results_ = _find_best_bandwidth(_data, min_bw, max_bw,
                    self.KernelDensity_args,
                    self.GridSearchCV_args,
                    verbose=verbose)
            self.best_estimator_ = self.search_results_.best_estimator_

            if verbose >= 2:
                print(self.best_estimator_.get_params())
        else:
            self.best_estimator_ = KernelDensity(**self.KernelDensity_args,
                    bandwidth = 1.06 * max_std / bw_factor)


    def predict(self, data, first_n_elements=None, n_parts=10, batch_size=32,
            recover_saved=False, verbose=0):
        """
        Параллельное вычисление оценки энтропии методом убрать-один-элемент.

        data             - выборка из исследуемой случайной величины.
        first_n_elements - оценка интеграла только по первым first_n_elements точкам.
                           (если параметр равен None, оценка производится по всем точкам)
        n_parts          - число частей, сохраняемых во временные файлы на случай
                           восстановления после сбоя.
        batch_size       - размер батча при распараллеливании.
        recover_saved    - произвести восстановление из временных файлов.
        verbose          - подробность вывода.
        """

        # Изменение точности оценщика.
        self.best_estimator_.set_params(rtol = self.estimation_rtol)

        # Нормировка данных.
        _data = data @ self.scaling_matrix_ if self.rescale else data

        # Функция для вычисления элемента суммы.
        def _loo_step(data, KernelDensity_params, index):
            loo_data = np.delete(data, index, axis=0)

            kde = KernelDensity()
            kde.set_params(**KernelDensity_params)
            kde.fit(loo_data)
            return kde.score_samples([data[index]])[0]

        # Создание временных папок для сохранения прогресса.
        path = os.path.abspath(os.getcwd())
        parts_path = path + "/.temp/LOO_PARTS/"
        os.makedirs(parts_path, exist_ok=True)

        # Если дано first_n_elements, энтропия будет оцениваться только на первых
        # first_n_elements элементах.
        n_elements = first_n_elements
        if n_elements is None:
            n_elements = len(_data)

        # Число частей и массив, их содержащий.
        n_elements_per_part = int(math.ceil(n_elements / n_parts))
        log_probs = []

        # Восстанавливаем прогресс, если требуется.
        n_recovered_parts = 0
        if recover_saved:
            for filename in os.listdir(parts_path):
                if filename.endswith(".csv"):
                    log_probs.append(np.loadtxt(parts_path + filename))
                    n_recovered_parts += 1

        if verbose >= 1:
            print("Восстановлено блоков данных: %d" % n_recovered_parts)

        # Подсчёт логарифма вероятности в точках.
        for part in range(n_recovered_parts, n_parts):
            log_probs.append(
                np.array(
                    Parallel(n_jobs = self.n_jobs, verbose = verbose, batch_size = batch_size)(
                        delayed(_loo_step)(_data, self.best_estimator_.get_params(), index) for index in range(
                            part * n_elements_per_part, min((part + 1) * n_elements_per_part, n_elements)
                        )
                    )
                )
            )
            np.savetxt(parts_path + str(part) + ".csv", log_probs[part], delimiter="\n")

        # Объединение в один массив.
        log_prob = np.concatenate(log_probs)

        # Суммирование и нахождение стандартного отклонения.
        average = -math.fsum(log_prob) / n_elements
        squared_deviations = np.zeros(n_elements)
        for index in range(n_elements):
            squared_deviations[index] = (log_prob[index] - average)**2
        standard_deviation = np.sqrt(math.fsum(squared_deviations) / (n_elements * (n_elements - 1)))

        # Удаление временных файлов.
        shutil.rmtree(path + '/.temp/')

        return average - self.scaling_delta_entropy_, standard_deviation