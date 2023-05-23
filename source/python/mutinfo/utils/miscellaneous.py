import math
import numpy as np


def minimize_recursive(function: callable, left: float, right: float,
                       n_points: int=7, atol: float=0.0, rtol: float=1e-2,
                       verbose: int=0) -> float:
    """
    Recursive function minimization.
    
    Parameters
    ----------
    function : callable
        Unimodal function.
    left : float
        The left end of the search interval.
    right : float
        The right end of the search interval.
    n_points : int
        Search grid size.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    verbose : int
        Output verbosity.
    """

    while True:
        grid = np.logspace(np.log10(left), np.log10(right), n_points)
        if verbose >= 1:
            print("Grid search: ", grid)
            
        # Evaluate the function along the grid.
        evaluated = np.array([function(element) for element in grid])
        best_index = np.nanargmin(evaluated)

        # Choose new interval.
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
    Volume of a multidimensional ball.
    
    Parameters
    ----------
    dim : int
        Dimension.
    radius : float
        Ball radius.
    """
    
    return ((math.sqrt(math.pi) * radius)**dim) / math.gamma(0.5 * dim + 1.0)