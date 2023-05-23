"""
Non-uniform Savitzky-Golay filter.
"""

import numpy as np


def nonuniform_savgol_filter(x, y, window_size: float, polyorder: int) -> np.array:
    """
    Non-uniform Savitzky-Golay filter.
    
    Parameters
    ----------
    x : array_like
        Data along "x" axis.
    y : array_like
        Data along "y" axis.
    window_size : float
        Size of the moving window.
    polyorder : int
        Order of fitted polynomials.
    """
    
    # Converting inputs.
    if not type(x) is np.array:
        x = np.array(x)
    if not type(y) is np.array:
        y = np.array(y)
        
    # Shape check.
    if len(x.shape) != 1:
        raise ValueError("x must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y arrays must be of the same shape")
    if np.any(x[:-1] > x[1:]):
        raise ValueError("x must be monotonically non-decreasing")
        
    delta = 0.5 * window_size
    
    polynomial = np.polynomial.polynomial.Polynomial(0)
    filtered_y = np.empty_like(x)
    for index in range(x.shape[0]):
        current_x = x[index]
        left  = np.searchsorted(x, current_x - delta)
        right = np.searchsorted(x, current_x + delta, side='right')
        
        n_points = right - left
        #print(f"{left} : {index} : {right};  {n_points}")
        current_polyorder = min(polyorder, n_points-1)
        
        polynomial = polynomial.fit(x[left:right], y[left:right], deg=current_polyorder)
        filtered_y[index] = polynomial(x[index])
        
    return filtered_y