import math
import numpy as np

from scipy.special import loggamma

from joblib import Parallel, delayed
from sklearn.neighbors import BallTree, DistanceMetric

from ..utils.miscellaneous import minimize_recursive, ball_volume


class Functional:
    """
    Класс для вычисление функционалов на основе оценки плотности.
    """
    
    def __init__(self, atol: float=0.0, rtol: float=0.0):
        """
        Инициализация экземпляра класса.
        
        Параметры
        ---------
        atol : float
            Допустимая абсолютная ошибка.
        rtol : float
            Допустимая относительная ошибка.
        """
        
        self.atol = atol
        self.rtol = rtol
    
    
    def fit(self, X, y=None, sample_weight=None):
        """
        Построить оценку плотности по данным.

        Параметры
        ---------
        X : array_like
            Данные образцов.
        y : array_like
            Данные меток (игнорируются).
        sample_weight : array_like
            Веса образцов.
        """
        
        self.data = X
        
        
    def get_densities(self, X):
        """
        Получение плотности в точках X.
        
        Параметры
        ---------
        X : array_like
            Набор точек.
        """
        
        raise NotImplementedError
        
        
    def get_loo_densities(self, outliers_atol: float=0.0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        Параметры
        ---------
        outliers_atol : float
            Абсолютный порог для значения плотности,
            ниже которого точки отбрасываются как выбросы.
        """
        
        raise NotImplementedError
        
        
    def integrate(self, func: callable, outliers_atol: float=0.0,
                  bootstrap_size: int=None, verbose: int=0):
        """
        Вычисление функционала методом убрать-один-элемент.
        
        Параметры
        ---------
        func : callable
            Интегрируемая функция.
        outliers_atol : float
            Абсолютный порог для значения плотности,
            ниже которого точки отбрасываются как выбросы.
        bootstrap_size : int
            Размер бустстрепной выборки.
        verbose : int
            Подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Получение плотностей.
        densities = self.get_loo_densities(outliers_atol)
        if densities is None:
            return np.nan, np.nan
        
        if bootstrap_size is None:
            # Вычисление функционала простым усреднением.
            values = self._get_values(func, densities)
            
            # Среднее и дисперсия функционала.
            mean = math.fsum(values) / n_samples
            std  = np.std(values) / np.sqrt(n_samples)
            
        else:
            # Вычисление функционала методом bootstrap.
            values = []
            for i in range(bootstrap_size):
                values.append(
                    math.fsum(self._get_values(func, np.random.choice(densities, size=n_samples)) / n_samples)
                )

            # Среднее и дисперсия функционала.      
            mean = np.mean(values)
            std  = np.std(values)

        return mean, std
    
    
    def _get_values(self, func: callable, densities):
        """
        Вычисление значений функции.
        
        Параметры
        ---------
        func : callable
            Интегрируемая функция.
        densities : array_like
            Вычисленные плотности.
        """
        
        # Если массив плотностей одномерен, добавляем фиктивную ось
        # - обобщение на взвешенный случай.
        if len(densities.shape) == 1:
            densities = densities[:,np.newaxis]
        
        # Веса.
        n_components = densities.shape[1]
        if not hasattr(self, 'weights'):
            weights = np.zeros(n_components)
            weights[0] = 1.0
        else:
            weights = self.weights
            
        # Вычисление значений.
        return func(densities) @ weights



class KDEFunctional(Functional):
    """
    Класс для вычисление функционалов на основе ядерной оценки плотности.
    """

    def __init__(self, *args, kernel: str='gaussian', bandwidth_algorithm: str='loo_cl',
                 tree_algorithm: str='ball_tree',
                 tree_params: dict={'leaf_size': 40, 'metric': 'euclidean'}, n_jobs: int=1):
        """
        Инициализация экземпляра класса.
        
        Параметры
        ---------
        kernel : str
            Ядро.
        bandwidth_algorithm : str
            Алгоритм выбора ширины окна.
              'loo_cl'  - перекрестная проверка методом убрать-один-элемент.
              'loo_lsq' - минимизация оценки среднеквадратического отклонения.
        tree_algorithm : str
            Используемое дерево.
        tree_params : dict
            Параметры дерева.
        n_jobs : int
            Число потоков, используемых при вычислении оценки.
        """
        
        super().__init__(*args)
        
        self.kernel = kernel
        self.bandwidth_algorithm = bandwidth_algorithm
        self.bandwidth = None
        self.tree_algorithm = tree_algorithm
        self.tree_params = tree_params
        self.n_jobs = n_jobs


    def fit(self, X, y=None, sample_weight=None,
            fit_bandwidth: bool=True, verbose: int=0):
        """
        Построить ядерную оценку плотности по данным.

        Параметры
        ---------
        X : array_like
            Данные образцов.
        y : array_like
            Данные меток (игнорируются).
        sample_weight : array_like
            Веса образцов (игнорируются).
        fit_bandwidth : bool
            Требуется ли подбирать ширину окна.
        verbose : int
            Подробность вывода.
        """

        if len(X.shape) != 2:
            raise TypeError("X must be of shape (?,?)")
            
        self.data = X
        
        if self.tree_algorithm == 'ball_tree':
            self.tree_ = BallTree(X, **self.tree_params)
        else:
            raise NotImplementedError
            
        # Выбор ширины окна.
        if fit_bandwidth:
            self.set_optimal_bandwidth(verbose=verbose)
            
            
    def get_loo_densities(self, outliers_atol: float=0.0, parallel: bool=True,
                          n_parts: int=None, verbose: int=0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        Параметры
        ---------
        outliers_atol : float
            Абсолютный порог для значения плотности,
            ниже которого точки отбрасываются как выбросы.
        parallel : bool
            Многопоточное вычисление плотностей.
        n_parts : int
            Число частей, на которые разбиваются данные при параллельной обработке.
        verbose : int
            Подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Вычисление значения плотности в центре ядра.
        if self.kernel == 'gaussian':
            diag_element = (1.0 / np.sqrt(2.0 * np.pi))**dim
            
        elif self.kernel == 'tophat':
            diag_element = 1.0 / ball_volume(dim)
            
        elif self.kernel == 'epanechnikov':
            diag_element = (0.5 * dim + 1.0) * math.gamma(0.5 * dim + 1.0) / np.power(np.pi, 0.5 * dim)
            
        else:
            raise NotImplementedError
            
        # Нормировка на dim-мерный шар.
        diag_element /= self.bandwidth**dim

        # Подсчёт плотностей вероятности в точках.
        if parallel:
            # Разбиение всего массива данных на части и параллельная обработка.
            if n_parts is None:
                n_parts = self.n_jobs
            n_samples_per_part = int(math.floor(n_samples / n_parts))
            
            # Функция для вычисления куска массива плотностей.
            def _loo_step(tree, bandwidth, begin, end, params):
                return tree.kernel_density(tree.data[begin:end,:], bandwidth, **params)

            # Параметры вычисления плотности.
            params = {
                'kernel'        : self.kernel,
                'atol'          : self.atol,
                'rtol'          : self.rtol,
                'breadth_first' : True, #self.breadth_first,
                'return_log'    : False
            }
            
            # Параллельное вычисление.
            densities = Parallel(n_jobs=min(n_parts, self.n_jobs), verbose=verbose, batch_size=1, max_nbytes=None)(
                    delayed(_loo_step)(
                        self.tree_,
                        self.bandwidth,
                        part * n_samples_per_part,
                        (part + 1) * n_samples_per_part if part + 1 < n_parts else n_samples,
                        params
                    ) for part in range(n_parts)
                )

            # Объединение в один массив.
            densities = np.concatenate(densities)
            
        else:
            # Однопоточная обработка.
            densities = self.tree_.kernel_density(
                self.tree_.data,
                self.bandwidth,
                kernel=self.kernel,
                atol=self.atol,
                rtol=self.rtol,
                breadth_first=True, #self.breadth_first,
                return_log=False
            )

        # Вычитание пллотности в центре ядра.
        densities -= diag_element
        
        # Удаление статистических выбросов.
        densities = densities[densities > outliers_atol]
        n_samples = len(densities)
        
        # Если осталось меньше двух точек, возвращаем None.
        if n_samples <= 1:
            return None
        
        # Нормировка.
        densities /= (n_samples - 1)
        
        return densities


    def set_optimal_bandwidth(self, min_bw: float=None, max_bw: float=None,
                              verbose: int=0):
        """
        Поиск оптимальной ширины окна.
        
        Параметры
        ---------
        min_bw : float
            Левый конец интервала поиска ширины окна.
        max_bw : float
            Правый конец интервала поиска ширины окна.
        verbose : int
            Подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Константы, необходимые для подбора начального приближения.
        bw_factor = np.power(n_samples, 0.2 / dim)
        std = np.std(self.tree_.data, axis=0)
        min_std = np.min(std)
        max_std = np.max(std)
        
        # Начальное приближение - Silverman's rule-of-thumb.
        if min_bw is None:
            min_bw =  0.5 * min_std / bw_factor
        if max_bw is None:
            max_bw = 1.06 * max_std / bw_factor
        
        if self.bandwidth_algorithm == 'loo_cl':
            """
            Минимизация расстояния Кульбака-Лейблера между ядерной оценкой и эмпирическим распределением.
            Эквивалентно максимизации функции правдоподобия методом убрать-один-элемент.
            """
            
            def function_(bandwidth):
                self.bandwidth = bandwidth
                mean, std = self.integrate(np.log)
                return -mean
            
            self.bandwidth = minimize_recursive(function_, min_bw, max_bw, verbose=verbose)
        
        elif self.bandwidth_algorithm == 'loo_lsq':
            """
            Минимизация оценки среднеквадратической ошибки.
            """
            
            # Допустимая ошибка.
            eps = 1e-8 / n_samples
            
            if self.kernel == 'gaussian':
                # Функция для вычисления кросс-корреляции ядер.
                correlation_func = lambda x, bandwidth : (1.0 / (2.0 * bandwidth * np.sqrt(np.pi)))**dim * \
                    np.exp(-x**2 / (4.0 * bandwidth**2))
                
                # Функция, дающая для заданной ошибки предельный радиус поиска соседей.
                radius_func = lambda bandwidth : np.sqrt( -np.log( eps * (2.0 * bandwidth * np.sqrt(np.pi))**dim ) ) * \
                    2.0 * bandwidth
                
            else:
                # Пока что метод работает только для гауссова ядра.
                raise NotImplementedError
                
            def function_(bandwidth):
                self.bandwidth = bandwidth
                
                # Линейное слагаемое - математическое ожидание оценки плотности.
                mean, std = self.integrate(np.vectorize(lambda x : x))
                linear_term = -2.0 * mean
                
                # Квадратичное слагаемое - сумма кросс-корреляций ядер.
                radius = radius_func(bandwidth)
                ind, dist = self.tree_.query_radius(self.tree_.data, radius, return_distance=True)
                squared_term = []
                for index in range(n_samples):
                    squared_term.append(math.fsum(correlation_func(dist[index], bandwidth)))
                squared_term = math.fsum(squared_term) / n_samples**2
                
                # Наивное вычисление.
                #squared_term = 0.0
                #for index in range(n_samples):
                #    for jndex in range(index, n_samples):
                #        squared_term += correlation_func(self.data[index] - self.data[jndex], bandwidth)
                #squared_term *= 2.0 / n_samples**2
                        
                return squared_term + linear_term
            
            self.bandwidth = minimize_recursive(function_, min_bw, max_bw, verbose=verbose)
            
        
        return self.bandwidth
    
    
class KLFunctional(Functional):
    """
    Класс для вычисление функционалов методом Козаченко Леоненко.
    """

    def __init__(self, *args, k_neighbours: int=5, tree_algorithm: str='ball_tree',
                 tree_params: dict={'leaf_size': 40, 'metric': 'euclidean'}, n_jobs: int=1):
        """
        Инициализация экземпляра класса.
        
        Параметры
        ---------
        k_neighbours : int
            Число ближайших соседей, по которым вычисляется оценка плотности.
        tree_algorithm : str
            Используемое дерево.
        tree_params : dict
            Параметры дерева.
        n_jobs : int
            Число потоков, используемых при вычислении оценки.
        """
        
        if k_neighbours <= 0:
            raise ValueError("Number of neighbours must be positive")
            
        super().__init__(*args)
        
        self.k_neighbours = k_neighbours
        self.tree_algorithm = tree_algorithm
        self.tree_params = tree_params
        self.n_jobs = n_jobs
        
        self.weights = np.zeros(self.k_neighbours)
        self.weights[0] = 1.0
        #self.weights = np.ones(self.k_neighbours) / self.k_neighbours
        

    def fit(self, X, y=None, sample_weight=None, fit_weights: bool=True,
            verbose: int=0):
        """
        Построить ядерную оценку плотности по данным.

        Параметры
        ---------
        X : array_like
            Данные образцов.
        y : array_like
            Данные меток (игнорируются).
        sample_weight : array_like
            Веса образцов (игнорируются).
        fit_weights : bool
            Требуется ли подбирать веса метода.
        verbose : int
            Подробность вывода.
        """

        if len(X.shape) != 2 or X.shape[0] < self.k_neighbours:
            raise TypeError("X must be of shape (?, >= k_neigbours)")
            
        self.data = X
        
        if self.tree_algorithm == 'ball_tree':
            self.tree_ = BallTree(X, **self.tree_params)
        else:
            raise NotImplementedError
            
        # Подбор весов.
        if fit_weights:
            self.set_optimal_weights(verbose=verbose)
        
        
    def get_loo_densities(self, outliers_atol: float=0.0, verbose: int=0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        Параметры
        ---------
        outliers_atol : float
            Абсолютный порог для значения плотности,
            ниже которого точки отбрасываются как выбросы.
        verbose : int
            Подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Получение _k_neighbours ближайших соседей.
        distances, indexes = self.tree_.query(self.tree_.data, self.k_neighbours + 1, return_distance=True)
        distances = distances[:,1:]
        
        # Плотности.
        unit_ball_volume = ball_volume(dim)
        
        #psi = np.array([sum(1/n for n in range(1, k - 1))] for k in range(self.k_neighbours)) - np.euler_gamma
        psi = np.zeros(self.k_neighbours)
        psi[0] = -np.euler_gamma
        for index in range(1, self.k_neighbours):
            psi[index] = psi[index - 1] + 1 /  index
            
        densities = np.exp(psi) / (unit_ball_volume * np.power(distances, dim))
        
        # Удаление статистических выбросов.
        #densities = densities[densities > outliers_atol]
        #n_samples = len(densities)
        
        # Если осталось меньше двух точек, возвращаем None.
        #if n_samples <= 1:
        #    return None
        
        # Нормировка.
        densities /= (n_samples - 1)
        
        return densities
    

    def set_optimal_weights(self, rcond: float=1e-6, zero_constraints: bool=True,
                            verbose: int=0):
        """
        Поиск оптимальных весов.
        
        Параметры
        ---------
        rcond: float
            Порог регуляризации при нахождении вектора весов.
        zero_constraints: bool
            Добавлять ли ограничения, зануляющие некоторые веса.
        verbose : int
            Подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        if dim <= 4:
            # В вырожденном случае используем стандартные веса.
            self.weights = np.zeros(self.k_neighbours)
            self.weights[0] = 1.0
            
        else:
            # Составляем линейное ограничение
            constraints = []

            # Ограничение - сумма единиц.
            constraints.append(np.ones(self.k_neighbours) / self.k_neighbours)

            # Ограничения с гамма-функциями.
            n_gamma_constraints = dim // 4
            for k in range(1, n_gamma_constraints + 1):
                constraints.append(
                    #np.array([math.gamma(j + 2*k / dim) / math.gamma(j) for j in range(1, self.k_neighbours + 1)])
                    np.exp(loggamma(np.arange(1, self.k_neighbours + 1) + 2 * k / dim) - \
                           loggamma(np.arange(1, self.k_neighbours + 1)))
                )
                constraints[-1] /= np.linalg.norm(constraints[-1])
                
            # Ограничение отдельных элементов.
            if zero_constraints:
                nonzero = set(i * self.k_neighbours // dim - 1 for i in range(1, dim + 1))
                for j in range(self.k_neighbours):
                    if not j in nonzero:
                        constraint = np.zeros(self.k_neighbours)
                        constraint[j] = 1.0
                        constraints.append(constraint)
                    
            constraints = np.vstack(constraints)
            
            # Правая часть.
            rhs = np.zeros(constraints.shape[0])
            rhs[0] = 1.0 / self.k_neighbours

            self.weights = np.linalg.lstsq(constraints, rhs, rcond=rcond)[0]
            #self.weights /= np.sum(self.weights)
        
        return self.weights