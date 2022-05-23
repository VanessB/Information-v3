import math
import numpy as np

from joblib import Parallel, delayed
from sklearn.neighbors import BallTree, DistanceMetric

from ..utils.miscellaneous import minimize_recursive, ball_volume


class Functional:
    """
    Класс для вычисление функционалов на основе оценки плотности.
    """
    
    def __init__(self):
        """
        Инициализация.
        """
        
        pass
    
    def fit(self, X, y=None, sample_weight=None):
        """
        Построить оценку плотности по данным.

        X - данные образцов.
        y - данные меток (игнорируются)
        sample_weight - веса образцов.
        """
        
        self.data = X
        
    def get_densities(self, X):
        """
        Получение плотности в точках X.
        
        X - набор точек.
        """
        
        raise NotImplementedError
        
    def get_loo_densities(self, outliers_atol=0.0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        outliers_atol - абсолютный порог для значения плотности,
                        ниже которого точки отбрасываются как выбросы.
        """
        
        raise NotImplementedError
        
        
    def integrate(self, func, outliers_atol=0.0, bootstrap_size=None, verbose=0):
        """
        Вычисление функционала методом убрать-один-элемент.
        
        func           - функционал.
        outliers_atol  - абсолютный порог для значения плотности,
                         ниже которого точки отбрасываются как выбросы.
        bootstrap_size - размер бустстрепной выборки.
        verbose        - подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Получение плотностей.
        densities = self.get_loo_densities(outliers_atol)
        if densities is None:
            return np.nan, np.nan
        
        if bootstrap_size is None:
            # Вычисление функционала простым усреднением.
            values = func(densities)
            
            # Среднее и дисперсия функционала.
            mean = math.fsum(values) / n_samples
            std  = np.std(values) / np.sqrt(n_samples)
            
        else:
            # Вычисление функционала методом bootstrap.
            values = []
            for i in range(bootstrap_size):
                values.append(
                    math.fsum(func(np.random.choice(densities, size=n_samples)) / n_samples)
                )

            # Среднее и дисперсия функционала.      
            mean = np.mean(values)
            std  = np.std(values)

        return mean, std



class KDEFunctional(Functional):
    """
    Класс для вычисление функционалов на основе ядерной оценки плотности.
    """

    def __init__(self, *args, bandwidth=1.0, algorithm='ball_tree', kernel='gaussian',
                 metric='euclidean', atol=0, rtol=0, breadth_first=True, leaf_size=40,
                 metric_params=None, n_jobs=1):
        """
        Инициализация.
        """
        
        super().__init__(*args)

        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.n_jobs = n_jobs


    def fit(self, X, y=None, sample_weight=None):
        """
        Построить ядерную оценку плотности по данным.

        X - данные образцов.
        y - данные меток (игнорируются)
        sample_weight - веса образцов.
        """

        assert len(X.shape) == 2
        self.data = X
        
        if self.algorithm == 'ball_tree':
            self.tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
        else:
            raise NotImplementedError
            
            
    def get_loo_densities(self, outliers_atol=0.0, parallel=True, n_parts=None, verbose=0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        outliers_atol - абсолютный порог для значения плотности,
                        ниже которого точки отбрасываются как выбросы.
        parallel      - многопоточное вычисление плотностей.
        n_parts       - число частей, на которые разбиваются данные при параллельной обработке.
        verbose       - подробность вывода.
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

            params = {
                #'bandwidth' : self.bandwidth,
                'kernel'    : self.kernel,
                'atol'      : self.atol,
                'rtol'      : self.rtol,
                'breadth_first' : self.breadth_first,
                'return_log' : False
            }
            
            densities = Parallel(n_jobs=min(n_parts, self.n_jobs), verbose=verbose, batch_size=1)(
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
                breadth_first=self.breadth_first,
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


    def set_optimal_bandwidth(self, algorithm='loo_cl', min_bw=None, max_bw=None, verbose=0):
        """
        Поиск оптимальной ширины окна.
        
        algorithm - алгоритм для поиска оптимальной ширины окна.
                    Доступные варианты:
                    1) loo_cl - минимизация оценки расстояния Кульбака-Лейблера
                                методом убрать-один-элемент.
                    2) loo_lsq - минимизация оценки среднеквадратической ошибки.
        min_bw    - левый конец интервала поиска ширины окна.
        max_bw    - правый конец интервала поиска ширины окна.
        verbose   - подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        bw_factor = np.power(n_samples, 0.2 / dim)
        std = np.std(self.tree_.data, axis=0)
        min_std = np.min(std)
        max_std = np.max(std)
        
        # Начальное приближение - Silverman's rule-of-thumb.
        if min_bw is None:
            min_bw =  0.5 * min_std / bw_factor
        if max_bw is None:
            max_bw = 1.06 * max_std / bw_factor
        
        if algorithm == 'loo_cl':
            """
            Минимизация расстояния Кульбака-Лейблера между ядерной оценкой и эмпирическим распределением.
            Эквивалентно максимизации функции правдоподобия методом убрать-один-элемент.
            """
            
            def function_(bandwidth):
                self.bandwidth = bandwidth
                mean, std = self.integrate(np.log)
                return -mean
            
            self.bandwidth = minimize_recursive(function_, min_bw, max_bw, verbose=verbose)
        
        elif algorithm == 'loo_lsq':
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
                raise NotImplementedError
                
            def function_(bandwidth):
                self.bandwidth = bandwidth
                mean, std = self.integrate(np.vectorize(lambda x : x))
                linear_term = -2.0 * mean
                
                radius = radius_func(bandwidth)
                ind, dist = self.tree_.query_radius(self.tree_.data, radius, return_distance=True)
                squared_term = []
                for index in range(n_samples):
                    squared_term.append(math.fsum(correlation_func(dist[index], bandwidth)))
                squared_term = math.fsum(squared_term) / n_samples**2
                
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

    def __init__(self, *args, k_neighbours=1, algorithm='ball_tree', metric='euclidean',
                 atol=0, rtol=0, breadth_first=True, leaf_size=40,
                 metric_params=None, n_jobs=1):
        """
        Инициализация.
        """
        
        super().__init__(*args)

        self.k_neighbours = k_neighbours
        self.algorithm = algorithm
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        

    def fit(self, X, y=None, sample_weight=None):
        """
        Построить оценку плотности по данным.

        X - данные образцов.
        y - данные меток (игнорируются)
        sample_weight - веса образцов.
        """

        assert len(X.shape) == 2
        assert X.shape[0] >= 2
        self.data = X
        
        if self.algorithm == 'ball_tree':
            self.tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
        else:
            raise NotImplementedError
        
        
    def get_loo_densities(self, outliers_atol=0.0, verbose=0):
        """
        Получение плотности в точках, на которых было произведено обучение.
        Применяется метод убрать-один-элемент.
        
        outliers_atol - абсолютный порог для значения плотности,
                        ниже которого точки отбрасываются как выбросы.
        verbose       - подробность вывода.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Число ближайших соседей.
        _k_neighbours = min(self.k_neighbours, n_samples - 1)
        
        # Получение _k_neighbours ближайших соседей.
        dist, ind = self.tree_.query(self.tree_.data, _k_neighbours + 1, return_distance=True)
        max_dist = dist[:,-1]
        
        # Плотности.
        unit_ball_volume = ball_volume(dim)
        psi_k = sum(1/n for n in range(1, _k_neighbours - 1)) - np.euler_gamma
        densities = np.exp(psi_k) / (unit_ball_volume * np.power(max_dist, dim))
        
        # Удаление статистических выбросов.
        densities = densities[densities > outliers_atol]
        n_samples = len(densities)
        
        # Если осталось меньше двух точек, возвращаем None.
        if n_samples <= 1:
            return None
        
        # Нормировка.
        densities /= (n_samples - 1)
        
        return densities