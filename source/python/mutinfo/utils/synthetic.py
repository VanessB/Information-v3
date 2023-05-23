import numpy as np
from scipy.special import ndtr
from .dependent_norm import multivariate_normal_from_MI

def normal_to_uniform(X: np.array) -> np.array:
    """
    Map Gaussian random vector with unit variance to a uniform
    distribution on the [0; 1]^dim.
    
    Parameters
    ----------
    X : numpy.array
        Samples from a multivariate normal distribution, shape: (?,dim).
    """

    return ndtr(X)


def normal_to_segment(X: np.array, min_length: float) -> np.array:
    """
    Map Gaussian random vector with unit variance to segments.
    The coordinates of the ends are distributed uniformly, preserving order.
    
    Parameters
    ----------
    X : numpy.array
        Samples from a multivariate normal distribution, shape: (?,2).
    min_length : float
        Minimum length of the segment.
    """
        
    if len(X.shape) != 2 or X.shape[1] != 2:
        raise TypeError("X array must have shape (?,2)")
        
    if not (0.0 <= min_length < 1.0):
        raise ValueError("min_length must be in [0.0, 1.0)")
        
    n_samples = X.shape[0]

    # Obtaining uniformly distributed samples.
    coords = normal_to_uniform(X)

    # The first number is the left end.
    # It is uniformly distributed from `0.0` to `1.0 - min_length`.
    coords[:,0] = (1.0 - min_length) * (1.0 - np.sqrt(1.0 - coords[:,0]))

    # The last number is the right end.
    # Given the left end, it is uniformly distributed from the left end plus
    # `min_length` to 1.0.
    coords[:,1] *= 1.0 - coords[:,0] - min_length
    coords[:,1] += coords[:,0] + min_length

    return coords


def normal_to_rectangle_coords(X: np.array, min_width: float=0.0, max_width: float=1.0,
                               min_height: float=0.0, max_height: float=1.0) -> np.array:
    """
    Map Gaussian random vector with unit variance to rectangle parameters.
    The coordinates of the corners are distributed uniformly, preserving order.
    
    Параметры
    ---------
    X : numpy.array
        Samples from a multivariate normal distribution, shape: (?,4).
    min_width : float
        Minimum width of the rectangle.
    max_width : float
        Maximum width of the rectangle.
    min_height : float
        Minimum height of the rectangle.
    max_height : float
        Maximum height of the rectangle.
    """

    if len(X.shape) != 2 or X.shape[1] != 4:
        raise TypeError("X array must have shape (?,4)")

    coords = np.zeros_like(X)
    coords[:,0:2] = normal_to_segment(X[:,0:2], min_width / max_width) * max_width
    coords[:,2:4] = normal_to_segment(X[:,2:4], min_height / max_height) * max_height

    return coords


def rectangle_coords_to_rectangles(coords: np.array, img_width: int, img_height: int) -> np.array:
    """
    Map coordinates of rectangles to rasterized images of rectangles.
    
    Parameters
    ----------
    coords : numpy.array
        Coordinates of rectangles, shape: (?,4).
    img_width : float
        Image width.
    img_height : float
        Image height.
    """

    if len(coords.shape) != 2 or coords.shape[1] != 4:
        raise TypeError("coords array must have shape (?,4)")
        
    n_samples = coords.shape[0]

    # Images generation.
    images = np.zeros((n_samples, img_width, img_height))
    for sample_index in range(n_samples):
        # The mapping must be piecewise smooth. To achieve this, we color each
        # pixel according to the covered area of the rectangle.
        floor_coords = np.floor(coords[sample_index]).astype(int)

        # Naive and slow.
        for x_index in range(floor_coords[0], floor_coords[1] + 1):
            for y_index in range(floor_coords[2], floor_coords[3] + 1):
                dx = min(coords[sample_index][1], x_index + 1) - max(coords[sample_index][0], x_index)
                dy = min(coords[sample_index][3], y_index + 1) - max(coords[sample_index][2], y_index)

                images[sample_index][x_index][y_index] = dx * dy

    return images


def params_to_2d_distribution(params: np.array, func: callable,
                              img_width: int, img_height: int) -> np.array:
    """
    Smooth 2D parametric function and corresponding parameters to rasterized
    images of 2D-plot.
    
    Parameters
    ----------
    params : numpy.array
        Samples of parameters.
    func : callable
        Parametric function, the graph of which is used to draw images.
    img_width : float
        Image width.
    img_height : float
        Image height.
    """
    
    n_samples = params.shape[0]
    
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, img_width), np.linspace(0.0, 1.0, img_height))
    X = X[np.newaxis,:]
    Y = Y[np.newaxis,:]
    
    images = func(X, Y, params)
                
    return images