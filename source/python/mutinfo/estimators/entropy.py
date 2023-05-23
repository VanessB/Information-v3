"""
Entropy estimators implementation.

Entropy estimation requires normalizing the data (optionaly), parameters
selection and numerical integration of the logarithm of
the probability density function.
"""

import math
import numpy as np

from .functional import KDEFunctional, KLFunctional
from ..utils.matrices import get_scaling_matrix, get_matrix_entropy


class EntropyEstimator:
    """
    Class for entopy estimation.
    """

    def __init__(self, rescale: bool=True, method: str='KDE',
                 functional_params: dict=None):
        """
        Initialization.
        
        Parameters
        ----------
        rescale : bool
            Enables data normalization step.
        method : str
            PDF estimation method
              'KDE' - kernel density estimation
              'KL'  - Kozachenko-Leonenko
        _functional_params : dict
            Additional parameters passed to the functional evaluator.
        """

        self.rescale = rescale
        self._scaling_matrix = None
        self._scaling_delta_entropy = 0.0
        
        if method == 'KDE':
            self._functional = KDEFunctional(**functional_params)
        elif method == 'KL':
            self._functional = KLFunctional(**functional_params)
        else:
            raise NotImplementedError(f"Method {method} is not implemented")


    def fit(self, data, fit_scaling_matrix: bool=True,
            verbose: int=0, **kwargs):
        """
        Data preprocessing and selection of functional evaluator parameters.

        Parameters
        ----------
        data : array_like
            I.i.d. samples from the random variable.
        fit_scaling_matrix : bool
            Fit matrix for data normalization.
        fit_bandwidth : bool
            Find optimal bandwidth (KDE only).
        verbose : int
            Output verbosity.
        """

        # Data normalization.
        if self.rescale:
            if fit_scaling_matrix:
                # Covariance matrix (required for normalization).
                # It is taken into account that in the case of one-dimensional data,
                # np.cov returns a number.
                cov_matrix = np.cov(data, rowvar=False)
                if data.shape[1] == 1:
                    cov_matrix = np.array([[cov_matrix]])
                
                # Getting the scaling matrix from the covariance matrix.
                self._scaling_matrix = get_scaling_matrix(cov_matrix)
                self._scaling_delta_entropy = -0.5 * get_matrix_entropy(cov_matrix)

            data = data @ self._scaling_matrix
            
        # Functional evaluator.
        self._functional.fit(data, **kwargs, verbose=verbose)

            
    def estimate(self, data, verbose: int=0) -> (float, float):
        """
        Entropy estimation.
        
        Parameters
        ----------
        verbose : int
            Output verbosity.
        """

        # The evaluation itself is performed by the functional evaluator.
        mean, std = self._functional.integrate(np.log, verbose=verbose)

        return -mean - self._scaling_delta_entropy, std