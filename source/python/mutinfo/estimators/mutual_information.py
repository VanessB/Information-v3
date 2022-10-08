import math
import numpy as np
from scipy.linalg import block_diag
from collections  import Counter
from .entropy import EntropyEstimator


class MutualInfoEstimator:
    """
    Класс-эстиматор взаимной информации.

    """

    def __init__(self, X_is_discrete: bool=False, Y_is_discrete: bool=False,
                 entropy_estimator_params: dict={'method': 'KL', 'functional_params': None}):
        """
        Инициализация.

        Параметры
        ---------
        X_is_discrete : bool
            Является ли X дискретной случайной величиной.
        Y_is_discrete : bool
            Является ли Y дискретной случайной величиной.
        entropy_estimator_params : dict
            Параметры оценщика энтропии.
        """

        self._X_is_discrete = X_is_discrete
        self._Y_is_discrete = Y_is_discrete
        self.entropy_estimator_params = entropy_estimator_params

        self._X_entropy_estimator = None
        self._Y_entropy_estimator = None
        self._X_Y_entropy_estimator = None


    def fit(self, X, Y, verbose: int=0):
        """
        Подгонка параметров эстиматора под экспериментальные данные.

        Параметры
        ---------
        X : Iterable
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same length")

        if not self._X_is_discrete and not self._Y_is_discrete:
            if self.entropy_estimator_params['method'] == 'KDE':
                if verbose >= 1:
                    print("Настройка оценщика для (X,Y)")

                self._X_Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_Y_entropy_estimator.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для X")

                self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_entropy_estimator.fit(X, fit_bandwidth=False, verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для Y")

                self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._Y_entropy_estimator.fit(Y, fit_bandwidth=False, verbose=verbose)

                # Использование подобранной ширины окна для оценки плотностей X и Y.
                bandwidth = self._X_Y_entropy_estimator._functional.bandwidth
                self._X_entropy_estimator._functional.bandwidth = bandwidth
                self._Y_entropy_estimator._functional.bandwidth = bandwidth
                
            else:
                if verbose >= 1:
                    print("Настройка оценщика для (X,Y)")

                self._X_Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_Y_entropy_estimator.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для X")

                self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_entropy_estimator.fit(X, verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для Y")

                self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._Y_entropy_estimator.fit(Y, verbose=verbose)
                
        elif self._X_is_discrete and not self._Y_is_discrete:
            if verbose >= 1:
                print("Настройка оценщика для Y")

            self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            self._Y_entropy_estimator.fit(Y, verbose=verbose)

        elif not self._X_is_discrete and self._Y_is_discrete:
            if verbose >= 1:
                print("Настройка оценщика для X")

            self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            self._X_entropy_estimator.fit(X, verbose=verbose)

        else:
            # В случае, когда обе случайные величины дискретные, подгонка не требуется.
            pass


    def estimate(self, X, Y, verbose: int=0):
        """
        Оценка взаимной информации.

        Параметры
        ---------
        X : Iterable
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same length")

        if not self._X_is_discrete and not self._Y_is_discrete:
            return self.estimate_cont_cont_(X, Y, verbose=verbose)

        elif self._X_is_discrete and not self._Y_is_discrete:
            return self.estimate_cont_disc_(Y, X, self._Y_entropy_estimator, verbose=verbose)

        elif not self._X_is_discrete and self._Y_is_discrete:
            return self.estimate_cont_disc_(X, Y, self._X_entropy_estimator, verbose=verbose)

        else:
            return self.estimate_disc_disc_(X, Y, verbose=verbose)


    def estimate_cont_cont_(self, X, Y, verbose: int=0):
        """
        Оценка в случае, когда обе случайные величины непрерывные.

        Параметры
        ---------
        X : array_like
            Выборка из первой случайной величины.
        Y : array_like
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        if verbose >= 1:
            print("Оценка энтропии для X")
        H_X, H_X_err = self._X_entropy_estimator.estimate(X, verbose=verbose)

        if verbose >= 1:
            print("Оценка энтропии для Y")
        H_Y, H_Y_err = self._Y_entropy_estimator.estimate(Y, verbose=verbose)

        if verbose >= 1:
            print("Оценка энтропии для (X,Y)")
        H_X_Y, H_X_Y_err = self._X_Y_entropy_estimator.estimate(np.concatenate([X, Y], axis=1), verbose=verbose)

        return (H_X + H_Y - H_X_Y, H_X_err + H_Y_err + H_X_Y_err)


    def estimate_cont_disc_(self, X, Y, X_entropy_estimator: EntropyEstimator,
                            verbose: int=0):
        """
        Оценка в случае, когда X непрерывен, а Y дискретен.
        
        Параметры
        ---------
        X : array_like
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        X_entropy_estimator : EntropyEstimator
            Оценщик энтропии для X.
        verbose : int
            Подробность вывода.
        """

        if verbose >= 1:
            print("Оценка энтропии для непрерывной случайной величины")
        H_X, H_X_err = X_entropy_estimator.estimate(X, verbose=verbose)

        # Подсчёт частот.
        frequencies = Counter(Y)
        for y in frequencies.keys():
            frequencies[y] /= Y.shape[0]

        if verbose >= 2:
            print("Частоты: ")
            print(frequencies)

        # Вычисление условной энтропии для каждого класса Y.
        H_X_mid_y = dict()
        for y in frequencies.keys():
            X_mid_y = X[Y == y]

            # Для каждого y требуется обучить отдельный оценщик.
            if verbose >= 1:
                print("Оценка энтропии для непрерывной случайной величины при условии дискретной")
            X_mid_y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            X_mid_y_entropy_estimator.fit(X_mid_y, verbose=verbose)
            #X_mid_y_entropy_estimator.best_estimator_.set_params(**X_entropy_estimator.best_estimator_.get_params())
            H_X_mid_y[y] = X_mid_y_entropy_estimator.estimate(X_mid_y, verbose=verbose)

        # Итоговая условная энтропия для X.
        cond_H_X     = math.fsum([frequencies[y] * H_X_mid_y[y][0] for y in frequencies.keys()])
        cond_H_X_err = math.fsum([frequencies[y] * H_X_mid_y[y][1] for y in frequencies.keys()])

        return (H_X - cond_H_X, H_X_err + cond_H_X_err)


    def estimate_disc_disc_(self, X, Y, verbose=0):
        """
        Оценка в случае, когда обе случайные величины дискретные.

        Параметры
        ---------
        X : Iterable
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        H_X = 0.0
        H_Y = 0.0
        H_X_Y = 0.0

        frequencies_X = Counter(X)
        for x in frequencies_X.keys():
            frequencies_X[x] /= X.shape[0]
            H_X -= frequencies_X[x] * np.log(frequencies_X[x])

        frequencies_Y = Counter(Y)
        for y in frequencies_Y.keys():
            frequencies_Y[y] /= Y.shape[0]
            H_Y -= frequencies_Y[y] * np.log(frequencies_Y[y])

        frequencies_X_Y = Counter(np.concatenate([X, Y], axis=1))
        for x_y in frequencies_X_Y.keys():
            frequencies_X_Y[x_y] /= X.shape[0]
            H_X_Y -= frequencies_X_Y[x_y] * np.log(frequencies_X_Y[x_y])

        if verbose >= 2:
            print("Частоты X: ")
            print(frequencies_X)

            print("Частоты Y: ")
            print(frequencies_Y)

            print("Частоты (X, Y): ")
            print(frequencies_X_Y)

        return (H_X + H_Y - H_X_Y, 0.0)



class LossyMutualInfoEstimator(MutualInfoEstimator):
    """
    Класс-эстиматор взаимной информации, использующий сжатие.
    """

    def __init__(self, X_compressor: callable=None, Y_compressor: callable=None,
                 *args, **kwargs):
        """
        Инициализация.

        Параметры
        ---------
        X_compressor : callable
            Вызываемый объект, выполняющий сжатие данных X.
        Y_compressor : callable
            Вызываемый объект, выполняющий сжатие данных Y.
        """

        super().__init__(*args, **kwargs)
        self._X_compressor = X_compressor
        self._Y_compressor = Y_compressor


    def fit(self, X, Y, verbose: int=0):
        """
        Подгонка параметров эстиматора под экспериментальные данные.

        Параметры
        ---------
        X : Iterable
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        X_compressed = X if self._X_compressor is None else self._X_compressor(X)
        Y_compressed = Y if self._Y_compressor is None else self._Y_compressor(Y)
        super().fit(X_compressed, Y_compressed, verbose=verbose)

        
    def estimate(self, X, Y, verbose: int=0):
        """
        Оценка взаимной информации.

        Параметры
        ---------
        X : Iterable
            Выборка из первой случайной величины.
        Y : Iterable
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        X_compressed = X if self._X_compressor is None else self._X_compressor(X)
        Y_compressed = Y if self._Y_compressor is None else self._Y_compressor(Y)
        return super().estimate(X_compressed, Y_compressed, verbose=verbose)