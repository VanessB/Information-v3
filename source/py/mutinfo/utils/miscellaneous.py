import math
import numpy as np


def minimize_recursive(function, left, right, n_points=7, atol=0.0, rtol=1e-2, verbose=0):
    """
    Рекурсивный поиск минимума функции.
    
    function - исследуемая унимодальная функция.
    left     - левый конец отрезка поиска.
    right    - правый конец отрезка поиска.
    n_points - размер сетки поиска.
    atol     - допустимое абсолютное значение ошибки.
    rtol     - допустимое относительное значение ошибки.
    verbose  - подробность вывода.
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
            right  = grid[1]
        elif best_index == n_points - 1:
            right *= right / left
            left   = grid[-2]
        else:
            left   = grid[best_index - 1]
            right  = grid[best_index + 1]

            if right - left < rtol * grid[best_index] + atol:
                return grid[best_index]
            
            
def ball_volume(dim, radius=1.0):
    """
    Объём многомерного шара.
    
    dim    - размерность пространства.
    radius - радиус шара.
    """
    
    return ((math.sqrt(math.pi) * radius)**dim) / math.gamma(0.5 * dim + 1.0)