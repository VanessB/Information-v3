import math
import numpy as np


def minimize_recursive(function: callable, left: float, right: float,
                       n_points: int=7, atol: float=0.0, rtol: float=1e-2,
                       verbose: int=0) -> float:
    """
    Рекурсивный поиск минимума функции.
    
    Параметры
    ---------
    function : callable
        Исследуемая унимодальная функция.
    left : float
        Левый конец отрезка поиска.
    right : float
        Правый конец отрезка поиска.
    n_points : int
        Размер сетки поиска.
    atol : float
        Допустимое абсолютное значение ошибки.
    rtol : float
        Допустимое относительное значение ошибки.
    verbose : int
        Подробность вывода.
    """

    while True:
        grid = np.logspace(np.log10(left), np.log10(right), n_points)
        if verbose >= 1:
            print("Поиск по сетке: ", grid)
            
        # Вычисление значений функции вдоль сетки.
        evaluated = np.array([function(element) for element in grid])
        best_index = np.nanargmin(evaluated)

        # Выбор нового отрезка.
        if best_index == 0:
            left  *= left / right
            right  = grid[2]
        elif best_index == n_points - 1:
            right *= right / left
            left   = grid[-3]
        else:
            left   = grid[best_index - 1]
            right  = grid[best_index + 1]

            if right - left < rtol * grid[best_index] + atol:
                return grid[best_index]
            
            
def ball_volume(dim: int, radius: float=1.0) -> float:
    """
    Объём многомерного шара.
    
    Параметры
    ---------
    dim : int
        Размерность пространства.
    radius : float
        Радиус шара.
    """
    
    return ((math.sqrt(math.pi) * radius)**dim) / math.gamma(0.5 * dim + 1.0)