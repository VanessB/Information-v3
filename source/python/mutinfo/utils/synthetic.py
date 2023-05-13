import numpy as np
from scipy.special import ndtr
from .dependent_norm import multivariate_normal_from_MI

def normal_to_uniform(X: np.array) -> np.array:
    """
    Гауссов случайный вектор с единичными дисперсиями в равномерное
    распределение на [0; 1]^dim.
    
    Параметры
    ---------
    X : numpy.array
        Выборка из многомерного нормального распределения размерности (?,dim).
    """

    return ndtr(X)


def normal_to_segment(X: np.array, min_length: float) -> np.array:
    """
    Гауссов случайный вектор с единичными дисперсиями в координаты концов отрезка.
    Координаты концов распределены равномерно с учётом необходимости сохранения порядка.
    
    Параметры
    ---------
    X : numpy.array
        Выборка из многомерного нормального распределения размерности (?,2).
    min_length : float
        Минимальная длина отрезка.
    """
        
    if len(X.shape) != 2 or X.shape[1] != 2:
        raise TypeError("X array must have shape (?,2)")
        
    if not (0.0 <= min_length < 1.0):
        raise ValueError("min_length must be in [0.0, 1.0)")
        
    n_samples = X.shape[0]

    # Получение равномерно распределённых сэмплов.
    coords = normal_to_uniform(X)

    # Первое число - координата левого конца.
    # Она должна быть распределена линейно от нуля до 1.0 - min_length
    coords[:,0] = (1.0 - min_length) * (1.0 - np.sqrt(1.0 - coords[:,0]))

    # Последнее число - координата правого конца.
    # При фиксированной первой координате она должна быть распределена
    # равномерно от координаты левого конца плюс min_length до 1.
    coords[:,1] *= 1.0 - coords[:,0] - min_length
    coords[:,1] += coords[:,0] + min_length

    return coords


def normal_to_rectangle_coords(X: np.array, min_width: float=0.0, max_width: float=1.0,
                               min_height: float=0.0, max_height: float=1.0) -> np.array:
    """
    Гауссов случайный вектор с единичными дисперсиями в координаты точек прямоугольника.
    Координаты точек распределены равномерно с учётом необходимости сохранения порядка.
    
    Параметры
    ---------
    X : numpy.array
        Выборка из многомерного нормального распределения размерности (?,4).
    min_width : float
        Минимальная ширина прямоугольника.
    max_width : float
        Максимальная ширина прямоугольника.
    min_height : float
        Минимальная высота прямоугольника.
    max_height : float
        Максимальная высота прямоугольника.
    """

    if len(X.shape) != 2 or X.shape[1] != 4:
        raise TypeError("X array must have shape (?,4)")

    coords = np.zeros_like(X)
    coords[:,0:2] = normal_to_segment(X[:,0:2], min_width / max_width) * max_width
    coords[:,2:4] = normal_to_segment(X[:,2:4], min_height / max_height) * max_height

    return coords


def rectangle_coords_to_rectangles(coords: np.array, img_width: int, img_height: int) -> np.array:
    """
    Координаты углов прямоугольников в изображения прямоугольников.
    
    Параметры
    ---------
    coords : numpy.array
        Выборка координат прямоугольников размерности (?,4).
    img_width : float
        Ширина изображения.
    img_height : float
        Высота изображения.
    """

    if len(coords.shape) != 2 or coords.shape[1] != 4:
        raise TypeError("coords array must have shape (?,4)")
        
    n_samples = coords.shape[0]

    # Непосредственная генерация прямоугольников.
    images = np.zeros((n_samples, img_width, img_height))
    for sample_index in range(n_samples):
        # Преобразование должно быть хотя бы кусочно-гладким.
        # Для этого каждый пиксель закрашиваем настолько, сколько в нём закрыто площади.
        floor_coords = np.floor(coords[sample_index]).astype(int)

        # Не самый оптимальный способ. Стоит переделать хотя бы заливку.
        for x_index in range(floor_coords[0], floor_coords[1] + 1):
            for y_index in range(floor_coords[2], floor_coords[3] + 1):
                dx = min(coords[sample_index][1], x_index + 1) - max(coords[sample_index][0], x_index)
                dy = min(coords[sample_index][3], y_index + 1) - max(coords[sample_index][2], y_index)

                images[sample_index][x_index][y_index] = dx * dy

    return images


def params_to_2d_distribution(params: np.array, func: callable,
                              img_width: int, img_height: int) -> np.array:
    """
    Распределение и параметры распределения в изображения распределения.
    
    Параметры
    ---------
    params : numpy.array
        Выборка параметров распределения.
    func : callable
        Функция, задающая распределение.
    img_width : float
        Ширина изображения.
    img_height : float
        Высота изображения.
    """
    
    n_samples = params.shape[0]
    
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, img_width), np.linspace(0.0, 1.0, img_height))
    X = X[None,:]
    Y = Y[None,:]
    
    images = func(X, Y, params)
    
    #images = np.zeros((n_samples, img_width, img_height))
    #for sample_index in range(n_samples):
    #    for x in range(img_width):
    #        for y in range(img_height):
    #            images[sample_index][x][y] = func(x / (img_width - 1), y / (img_height - 1), params[sample_index])
                
    return images