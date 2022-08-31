import math
import numpy as np
from scipy.linalg import block_diag
from collections  import Counter
from .entropy import EntropyEstimator


class MutualInfoEstimator:
    """
    Класс-эстиматор взаимной информации.

    """

    def __init__(self, X_is_discrete=False, Y_is_discrete=False,
                 entropy_estimator_params={'method': 'KL', 'functional_params': None}):
        """
        Инициализация.

        Параметры
        ---------
        X_is_discrete : bool
            Является ли X дискретной случайной величиной.
        Y_is_discrete : bool
            Является ли Y дискретной случайной величиной.
        n_jobs : int
            Число задействованных потоков.
        """

        self.X_is_discrete_ = X_is_discrete
        self.Y_is_discrete_ = Y_is_discrete
        self.entropy_estimator_params = entropy_estimator_params

        self.X_entropy_estimator_ = None
        self.Y_entropy_estimator_ = None
        self.X_Y_entropy_estimator_ = None


    def fit(self, X, Y, verbose=0):
        """
        Подгонка параметров эстиматора под экспериментальные данные.

        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        assert X.shape[0] == Y.shape[0]

        if not self.X_is_discrete_ and not self.Y_is_discrete_:
            if self.entropy_estimator_params['method'] == 'KDE':
                if verbose >= 1:
                    print("Настройка оценщика для (X,Y)")

                self.X_Y_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.X_Y_entropy_estimator_.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для X")

                self.X_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.X_entropy_estimator_.fit(X, fit_bandwidth=False, verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для Y")

                self.Y_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.Y_entropy_estimator_.fit(Y, fit_bandwidth=False, verbose=verbose)

                # Использование подобранной ширины окна для оценки плотностей X и Y.
                bandwidth = self.X_Y_entropy_estimator_.functional_.bandwidth
                self.X_entropy_estimator_.functional_.bandwidth = bandwidth
                self.Y_entropy_estimator_.functional_.bandwidth = bandwidth
                
            else:
                if verbose >= 1:
                    print("Настройка оценщика для (X,Y)")

                self.X_Y_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.X_Y_entropy_estimator_.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для X")

                self.X_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.X_entropy_estimator_.fit(X, verbose=verbose)

                if verbose >= 1:
                    print("Настройка оценщика для Y")

                self.Y_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
                self.Y_entropy_estimator_.fit(Y, verbose=verbose)
                
        elif self.X_is_discrete_ and not self.Y_is_discrete_:
            if verbose >= 1:
                print("Настройка оценщика для Y")

            self.Y_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
            self.Y_entropy_estimator_.fit(Y, verbose=verbose)

        elif not self.X_is_discrete_ and self.Y_is_discrete_:
            if verbose >= 1:
                print("Настройка оценщика для X")

            self.X_entropy_estimator_ = EntropyEstimator(**self.entropy_estimator_params)
            self.X_entropy_estimator_.fit(X, verbose=verbose)

        else:
            # В случае, когда обе случайные величины дискретные, подгонка не требуется.
            pass


    def estimate(self, X, Y, verbose=0):
        """
        Оценка взаимной информации.

        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        assert X.shape[0] == Y.shape[0]

        if not self.X_is_discrete_ and not self.Y_is_discrete_:
            return self.estimate_cont_cont_(X, Y, verbose=verbose)

        elif self.X_is_discrete_ and not self.Y_is_discrete_:
            return self.estimate_cont_disc_(Y, X, self.Y_entropy_estimator_, verbose=verbose)

        elif not self.X_is_discrete_ and self.Y_is_discrete_:
            return self.estimate_cont_disc_(X, Y, self.X_entropy_estimator_, verbose=verbose)

        else:
            return self.estimate_disc_disc_(X, Y, verbose=verbose)


    def estimate_cont_cont_(self, X, Y, verbose=0):
        """
        Оценка в случае, когда обе случайные величины непрерывные.

        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        if verbose >= 1:
            print("Оценка энтропии для X")
        H_X, H_X_err = self.X_entropy_estimator_.estimate(X, verbose=verbose)

        if verbose >= 1:
            print("Оценка энтропии для Y")
        H_Y, H_Y_err = self.Y_entropy_estimator_.estimate(Y, verbose=verbose)

        if verbose >= 1:
            print("Оценка энтропии для (X,Y)")
        H_X_Y, H_X_Y_err = self.X_Y_entropy_estimator_.estimate(np.concatenate([X, Y], axis=1), verbose=verbose)

        return (H_X + H_Y - H_X_Y, H_X_err + H_Y_err + H_X_Y_err)


    def estimate_cont_disc_(self, X, Y, X_entropy_estimator, verbose=0):
        """
        Оценка в случае, когда X непрерывен, а Y дискретен.
        
        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
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
        X : array
            Выборка из первой случайной величины.
        Y : array
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

    def __init__(self, X_compressor=None, Y_compressor=None, *args, **kwargs):
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
        self.X_compressor_ = X_compressor
        self.Y_compressor_ = Y_compressor


    def fit(self, X, Y, verbose=0):
        """
        Подгонка параметров эстиматора под экспериментальные данные.

        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        X_compressed = X if self.X_compressor_ is None else self.X_compressor_(X)
        Y_compressed = Y if self.Y_compressor_ is None else self.Y_compressor_(Y)
        super().fit(X_compressed, Y_compressed, verbose=verbose)

        
    def estimate(self, X, Y, verbose=0):
        """
        Оценка взаимной информации.

        Параметры
        ---------
        X : array
            Выборка из первой случайной величины.
        Y : array
            Выборка из второй случайной величины.
        verbose : int
            Подробность вывода.
        """

        X_compressed = X if self.X_compressor_ is None else self.X_compressor_(X)
        Y_compressed = Y if self.Y_compressor_ is None else self.Y_compressor_(Y)
        return super().estimate(X_compressed, Y_compressed, verbose=verbose)