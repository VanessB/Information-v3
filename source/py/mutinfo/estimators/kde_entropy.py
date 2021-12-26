import csv
import json
import math
import numpy as np
import os
import shutil

from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


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

    def __init__(self, KernelDensity_args = {"kernel": "gaussian", "rtol": 1e-3},
            GridSearchCV_args  = {"cv": 5},
            n_jobs = 1, estimation_rtol = 0.0):

        self.KernelDensity_args = KernelDensity_args
        self.GridSearchCV_args  = GridSearchCV_args

        self.n_jobs = n_jobs
        self.GridSearchCV_args["n_jobs"] = self.n_jobs
        self.estimation_rtol = estimation_rtol

        self.search_results_ = None
        self.best_estimator_ = None


    def fit(self, data, verbose=0):
        """
        Подбор параметров ядерной оценки плотности по данным.

        data - выборка из исследуемой случайной величины.
        verbose - подробность вывода.
        """

        #std = np.max(np.std(data, axis = 0))
        cov_matrix = np.cov(data, rowvar=False)
        if data.shape[1] == 1:
            cov_matrix = np.array([[cov_matrix]])

        eigen_values, _ = np.linalg.eigh(cov_matrix)
        factor = np.power(data.shape[0], 0.2 / data.shape[1])

        min_bw = 0.5 * np.sqrt(eigen_values[0]) / factor
        max_bw = 1.06 * np.sqrt(eigen_values[-1]) / factor

        self.search_results_ = _find_best_bandwidth(data, min_bw, max_bw,
                self.KernelDensity_args,
                self.GridSearchCV_args,
                verbose=verbose)
        self.best_estimator_ = self.search_results_.best_estimator_

        if verbose >= 2:
            print(self.best_estimator_.get_params())


    def predict(self, data, first_N = None, parts = 10, recover_saved = False, verbose=0):
        """
        Параллельное вычисление оценки энтропии методом убрать-один-элемент.

        data          - выборка из исследуемой случайной величины.
        first_N       - оценка интеграла только по первым first_N точкам.
                        (если параметр равен None, оценка производится по всем точкам)
        parts         - число частей, сохраняемых во временные файлы на случай восстановления после сбоя.
        recover_saved - произвести восстановление из временных файлов.
        verbose       - подробность вывода.
        """

        self.best_estimator_.set_params(rtol = self.estimation_rtol)

        def _loo_step(data, KernelDensity_params, i):
            loo_data = data
            np.delete(loo_data, i)

            kde = KernelDensity()
            kde.set_params(**KernelDensity_params)
            kde.fit(loo_data)
            return kde.score_samples([data[i]])[0]

        # Создание временных папок для сохранения прогресса.
        path = os.path.abspath(os.getcwd())
        parts_path = path + "/.temp/LOO_PARTS/"
        os.makedirs(parts_path, exist_ok=True)

        # Если дано first_N, энтропия будет оцениваться только на первых first_N элементах.
        N = 0
        if first_N is None:
            N = len(data)
        else:
            N = first_N

        # Число частей и массив, их содержащий.
        N_per_part = int(math.ceil(N / parts))
        log_probs = []

        # Восстанавливаем прогресс, если требуется.
        recovered_parts = 0
        if recover_saved:
            for filename in os.listdir(parts_path):
                if filename.endswith(".csv"):
                    log_probs.append(np.loadtxt(parts_path + filename))
                    recovered_parts += 1

        if verbose >= 1:
            print("Восстановлено блоков данных: %d" % recovered_parts)

        # Подсчёт логарифма вероятности в точках.
        for part in range(recovered_parts, parts):
            log_probs.append(
                np.array(
                    Parallel(n_jobs = self.n_jobs, verbose = verbose, batch_size = 8)(
                        delayed(_loo_step)(data, self.best_estimator_.get_params(), i) for i in range(part * N_per_part, min((part + 1) * N_per_part, N))
                    )
                )
            )
            np.savetxt(parts_path + str(part) + ".csv", log_probs[part], delimiter="\n")

        # Объединение в один массив.
        log_prob = np.concatenate(log_probs)

        # Суммирование и нахождение стандартного отклонения.
        average = -math.fsum(log_prob) / N
        squared_deviations = np.zeros(N)
        for i in range(N):
            squared_deviations[i] = (log_prob[i] - average)**2
        standard_deviation = np.sqrt(math.fsum(squared_deviations) / (N * (N - 1)))

        # Удаление временных файлов.
        shutil.rmtree(parts_path)

        return average, standard_deviation
