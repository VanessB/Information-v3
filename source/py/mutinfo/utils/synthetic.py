import numpy as np
from scipy.special import ndtr
from .dependent_norm import multivariate_normal_from_MI

def normal_to_uniform(X):
    """
    Гауссов случайный вектор с единичными дисперсиями в равномерное
    распределение на [0; 1]^dim.
    """

    return ndtr(X)


def normal_to_rectangle_coords(X, img_width, img_height):
    """
    Гауссов случайный вектор с единичными дисперсиями в координаты углов прямоугольников.
    Координаты углов распределены равномерно с учётом необходимости сохранения порядка
    (левый верхний угол не ниже и не правее нежнего правого).
    """

    assert len(X.shape) == 2
    assert X.shape[1] == 4
    n_samples = X.shape[0]

    # Получение равномерно распределённых сэмплов.
    coords = normal_to_uniform(X)

    # Первые два числа - координаты левого верхнего угла.
    # Они должны быть распределены линейно от нуля до img_width/img_height
    coords[:,0:2] = 1.0 - np.sqrt(1.0 - coords[:,0:2])
    coords[:,0]  *= img_width
    coords[:,1]  *= img_height

    # Последние два числа - координаты правого нижнего угла.
    # При фиксированных первых двух координатах они должны быть распределены
    # равномерно от соответствующей координаты левого верхнего угла
    # до img_width/img_height
    coords[:,2]   *= img_width  - coords[:,0]
    coords[:,3]   *= img_height - coords[:,1]
    coords[:,2:4] += coords[:,0:2]

    return coords


def rectangle_coords_to_rectangles(coords, img_width, img_height):
    """
    Координаты углов прямоугольников в изображения прямоугольников.
    """

    assert len(coords.shape) == 2
    assert coords.shape[1] == 4
    n_samples = coords.shape[0]

    # Непосредственная генерация прямоугольников.
    images = np.zeros((n_samples, img_width, img_height))
    for sample_index in range(n_samples):
        # Преобразование должно быть хотя бы кусочно-гладким.
        # Для этого каждый пиксель закрашиваем настолько, сколько в нём закрыто площади.
        floor_coords = np.floor(coords[sample_index]).astype(int)

        # Не самый оптимальный способ. Стоит переделать хотя бы заливку.
        for x_index in range(floor_coords[0], floor_coords[2] + 1):
            for y_index in range(floor_coords[1], floor_coords[3] + 1):
                dx = min(coords[sample_index][2], x_index + 1) - max(coords[sample_index][0], x_index)
                dy = min(coords[sample_index][3], y_index + 1) - max(coords[sample_index][1], y_index)

                images[sample_index][x_index][y_index] = dx * dy

    return images