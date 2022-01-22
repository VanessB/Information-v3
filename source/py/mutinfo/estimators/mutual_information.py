import numpy as np
from scipy.linalg import block_diag
from collections  import Counter
from .kde_entropy import EntropyEstimator


class MutualInfoEstimator:
    """
    Класс-эстиматор взаимной информации.

    """

    def __init__(self, Y_is_discrette=False, n_jobs=1):
        """
        Инициализация.

        Y_is_discrette - является ли Y дискретной случайной величиной?
        n_jobs         - число задействованных потоков.
        """

        self.Y_is_discrette_ = Y_is_discrette
        self.n_jobs = n_jobs

        self.X_entropy_estimator_ = None
        self.Y_entropy_estimator_ = None
        self.X_Y_entropy_estimator_ = None


    def fit(self, X, Y, verbose=0):
        """
        Подгонка параметров эстиматора под экспериментальные данные.

        X - выборка из первой случайной величины.
        Y - выборка из второй случайной величины.
        verbose - подробность вывода.
        """

        assert X.shape[0] == Y.shape[0]

        if self.Y_is_discrette_:
            if verbose >= 1:
                print("Настройка оценщика для X")

            self.X_entropy_estimator_ = EntropyEstimator(n_jobs=self.n_jobs)
            self.X_entropy_estimator_.fit(X, verbose=verbose)

        else:

            if verbose >= 1:
                print("Настройка оценщика для (X,Y)")

            self.X_Y_entropy_estimator_ = EntropyEstimator(n_jobs=self.n_jobs)
            self.X_Y_entropy_estimator_.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

            if verbose >= 1:
                print("Настройка оценщика для X")

            self.X_entropy_estimator_ = EntropyEstimator(n_jobs=self.n_jobs)
            self.X_entropy_estimator_.fit(X, fit_bandwidth=False, verbose=verbose)

            if verbose >= 1:
                print("Настройка оценщика для Y")

            self.Y_entropy_estimator_ = EntropyEstimator(n_jobs=self.n_jobs)
            self.Y_entropy_estimator_.fit(Y, fit_bandwidth=False, verbose=verbose)

            # Использование подобранной ширины окна для оценки плотностей X и Y.
            bandwidth = self.X_Y_entropy_estimator_.best_estimator_.get_params()['bandwidth']
            self.X_entropy_estimator_.best_estimator_.set_params(bandwidth=bandwidth)
            self.Y_entropy_estimator_.best_estimator_.set_params(bandwidth=bandwidth)


    def predict(self, X, Y, verbose=0):
        """
        Оценка взаимной информации.

        X - выборка из первой случайной величины.
        Y - выборка из второй случайной величины.
        verbose - подробность вывода.
        """

        assert X.shape[0] == Y.shape[0]

        if verbose >= 1:
            print("Оценка энтропии для X")
        H_X, H_X_err = self.X_entropy_estimator_.predict(X, verbose=verbose)

        if self.Y_is_discrette_:
            # Подсчёт частот.
            frequencies = Counter(Y)
            for y in frequencies.keys():
                frequencies[y] /= X.shape[0]

            # Вычисление условной энтропии для каждого класса Y.
            H_X_mid_y = dict()
            for y in frequencies.keys():
                X_mid_y = np.array([X[i] for i in range(X.shape[0]) if Y[i] == y])

                H_X_mid_y[y] = self.X_entropy_estimator_.predict(X_mid_y, verbose=verbose)

            # Итоговая условная энтропия для X.
            cond_H_X     = np.sum([frequencies[y] * H_X_mid_y[y][0] for y in frequencies.keys()])
            cond_H_X_err = np.sum([frequencies[y] * H_X_mid_y[y][1] for y in frequencies.keys()])

            return (H_X - cond_H_X, H_X_err + cond_H_X_err)

        else:
            if verbose >= 1:
                print("Оценка энтропии для Y")
            H_Y, H_Y_err = self.Y_entropy_estimator_.predict(Y, verbose=verbose)
            if verbose >= 1:
                print("Оценка энтропии для (X,Y)")
            H_X_Y, H_X_Y_err = self.X_Y_entropy_estimator_.predict(np.concatenate([X, Y], axis=1), verbose=verbose)

            return (H_X + H_Y - H_X_Y, H_X_err + H_Y_err + H_X_Y_err)