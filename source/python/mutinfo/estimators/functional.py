import math
import numpy as np

from scipy.special import loggamma

from joblib import Parallel, delayed
from sklearn.neighbors import BallTree, DistanceMetric

from ..utils.miscellaneous import minimize_recursive, ball_volume


class Functional:
    """
    Class for evaluating functional based on density estimates.
    """
    
    def __init__(self, atol: float=0.0, rtol: float=0.0):
        """
        Initialization.
        
        Parameters
        ----------
        atol : float
            Absolute tolerance.
        rtol : float
            Relative tolerance.
        """
        
        self.atol = atol
        self.rtol = rtol
    
    
    def fit(self, X, y=None, sample_weight=None):
        """
        Build a density estimate from the data.

        Parameters
        ----------
        X : array_like
            I.i.d. samples.
        y : array_like
            Target data (ignored).
        sample_weight : array_like
            Samples weights.
        """
        
        self.data = X
        
        
    def get_densities(self, X):
        """
        Obtaining the density estimate at points X.
        
        Parameters
        ----------
        X : array_like
            I.i.d. samples.
        """
        
        raise NotImplementedError
        
        
    def get_loo_densities(self, outliers_atol: float=0.0):
        """
        Obtaining the density at the points on which the fitting was performed.
        The leave-one-out method is applied.
        
        Parameters
        ----------
        outliers_atol : float
            Absolute tolerance for the density value,
            below which points are discarded as outliers.
        """
        
        raise NotImplementedError
        
        
    def integrate(self, func: callable, outliers_atol: float=0.0,
                  bootstrap_size: int=None, verbose: int=0):
        """
        Functional evaluation according to the leave-one-out method.
        
        Parameters
        ----------
        func : callable
            Integrated function.
        outliers_atol : float
            Absolute tolerance for the density value,
            below which points are discarded as outliers.
        bootstrap_size : int
            Booststrap sample size.
        verbose : int
            Output verbosity.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Obtain density values.
        densities = self.get_loo_densities(outliers_atol)
        if densities is None:
            return np.nan, np.nan
        
        if bootstrap_size is None:
            # Functional evaluation by simple averaging.
            values = self._get_values(func, densities)
            
            # The mean and variance of the functional.
            mean = math.fsum(values) / n_samples
            std  = np.std(values) / np.sqrt(n_samples)
            
        else:
            # Functional evaluation using the bootstrap method.
            values = []
            for i in range(bootstrap_size):
                values.append(
                    math.fsum(self._get_values(func, np.random.choice(densities, size=n_samples)) / n_samples)
                )

            # The mean and variance of the functional.
            mean = np.mean(values)
            std  = np.std(values)

        return mean, std
    
    
    def _get_values(self, func: callable, densities):
        """
        Calculation of function values.
        
        Parameters
        ----------
        func : callable
            Integrated function.
        densities : array_like
            Density function values at corresponding points.
        """
        
        # If the density array is one-dimensional, add a dummy axis
        # - generalization to the weighted case.
        if len(densities.shape) == 1:
            densities = densities[:,np.newaxis]
        
        # Weights.
        n_components = densities.shape[1]
        if not hasattr(self, 'weights'):
            weights = np.zeros(n_components)
            weights[0] = 1.0
        else:
            weights = self.weights
            
        # Evaluation.
        return func(densities) @ weights



class KDEFunctional(Functional):
    """
    Class for evaluating functional based on kernel density estimate.
    """

    def __init__(self, *args, kernel: str='gaussian', bandwidth_algorithm: str='loo_ml',
                 tree_algorithm: str='ball_tree',
                 tree_params: dict={'leaf_size': 40, 'metric': 'euclidean'}, n_jobs: int=1):
        """
        Initialization
        
        Parameters
        ----------
        kernel : str
            Kernel of the mixture.
        bandwidth_algorithm : str
            Algorithm for selecting the bandwidth.
              'loo_ml'  - leave-one-out maximum likelihood.
              'loo_lsq' - least squares error.
        tree_algorithm : str
            Metric tree used for density estimation.
        tree_params : dict
            Metric tree parameters.
        n_jobs : int
            Number of jobs.
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
        Build a kernel density estimate.

        Parameters
        ----------
        X : array_like
            I.i.d. samples.
        y : array_like
            Target data (ignored).
        sample_weight : array_like
            Samples weights (ignored).
        fit_bandwidth : bool
            Do the bandwidth selection.
        verbose : int
            Output verbosity.
        """

        if len(X.shape) != 2:
            raise TypeError("X must be of shape (?,?)")
            
        self.data = X
        
        if self.tree_algorithm == 'ball_tree':
            self.tree_ = BallTree(X, **self.tree_params)
        else:
            raise NotImplementedError
            
        # Bandwidth selection.
        if fit_bandwidth:
            self.set_optimal_bandwidth(verbose=verbose)
            
            
    def get_loo_densities(self, outliers_atol: float=0.0, parallel: bool=True,
                          n_parts: int=None, verbose: int=0):
        """
        Obtaining the density at the points on which the fitting was performed.
        The leave-one-out method is applied.
        
        Parameters
        ----------
        outliers_atol : float
            Absolute tolerance for the density value,
            below which points are discarded as outliers.
        parallel : bool
            Multithreaded calculation of densities.
        n_parts : int
            Number of parts into which the data is divided during parallel processing.
        verbose : int
            Output verbosity.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Density value at the center of the kernel.
        if self.kernel == 'gaussian':
            diag_element = (1.0 / np.sqrt(2.0 * np.pi))**dim
            
        elif self.kernel == 'tophat':
            diag_element = 1.0 / ball_volume(dim)
            
        elif self.kernel == 'epanechnikov':
            diag_element = (0.5 * dim + 1.0) * math.gamma(0.5 * dim + 1.0) / np.power(np.pi, 0.5 * dim)
            
        else:
            raise NotImplementedError
            
        # Norming on a `dim`-dimensional ball.
        diag_element /= self.bandwidth**dim

        # Estimation of probability densities at points.
        if parallel:
            # Partitioning the whole data array into parts and parallel processing.
            if n_parts is None:
                n_parts = self.n_jobs
            n_samples_per_part = int(math.floor(n_samples / n_parts))
            
            # A function for calculating a slice of an array of density values.
            def _loo_step(tree, bandwidth, begin, end, params):
                return tree.kernel_density(tree.data[begin:end,:], bandwidth, **params)

            # Parameters for density estimation.
            params = {
                'kernel'        : self.kernel,
                'atol'          : self.atol,
                'rtol'          : self.rtol,
                'breadth_first' : True, #self.breadth_first,
                'return_log'    : False
            }
            
            # Parallel estimation.
            densities = Parallel(n_jobs=min(n_parts, self.n_jobs), verbose=verbose, batch_size=1, max_nbytes=None)(
                    delayed(_loo_step)(
                        self.tree_,
                        self.bandwidth,
                        part * n_samples_per_part,
                        (part + 1) * n_samples_per_part if part + 1 < n_parts else n_samples,
                        params
                    ) for part in range(n_parts)
                )

            # Merge into single array.
            densities = np.concatenate(densities)
            
        else:
            # Single-threaded processing.
            densities = self.tree_.kernel_density(
                self.tree_.data,
                self.bandwidth,
                kernel=self.kernel,
                atol=self.atol,
                rtol=self.rtol,
                breadth_first=True, #self.breadth_first,
                return_log=False
            )

        # Subtraction of the density at the center of the kernel.
        densities -= diag_element
        
        # Removing statistical outliers.
        densities = densities[densities > outliers_atol]
        n_samples = len(densities)
        
        # If there are fewer than two points left, return None.
        if n_samples <= 1:
            return None
        
        # Normalization.
        densities /= (n_samples - 1)
        
        return densities


    def set_optimal_bandwidth(self, min_bw: float=None, max_bw: float=None,
                              verbose: int=0):
        """
        Optimal bandwidth selection.
        
        Parameters
        ----------
        min_bw : float
            Minimum bandwidth.
        max_bw : float
            Maximum bandwidth.
        verbose : int
            Output verbosity.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Constants needed to select the initial guess.
        bw_factor = np.power(n_samples, 0.2 / dim)
        std = np.std(self.tree_.data, axis=0)
        min_std = np.min(std)
        max_std = np.max(std)
        
        # Initial guess - Silverman's rule-of-thumb.
        if min_bw is None:
            min_bw =  0.5 * min_std / bw_factor
        if max_bw is None:
            max_bw = 1.06 * max_std / bw_factor
        
        if self.bandwidth_algorithm == 'loo_ml':
            """
            Minimization of the Kullback-Leibler distance between the kernel estimate and the empirical distribution.
            Equivalent to the maximization of the likelihood function by the leave-one-out method.
            """
            
            def function_(bandwidth):
                self.bandwidth = bandwidth
                mean, std = self.integrate(np.log)
                return -mean
            
            self.bandwidth = minimize_recursive(function_, min_bw, max_bw, verbose=verbose)
        
        elif self.bandwidth_algorithm == 'loo_lsq':
            """
            Least squares method.
            """
            
            # Tolerance.
            eps = 1e-8 / n_samples
            
            if self.kernel == 'gaussian':
                # Function for calculating cross-correlation of kernels.
                correlation_func = lambda x, bandwidth : (1.0 / (2.0 * bandwidth * np.sqrt(np.pi)))**dim * \
                    np.exp(-x**2 / (4.0 * bandwidth**2))
                
                # The function that gives for a given tolerance a radius of search for neighbors.
                radius_func = lambda bandwidth : np.sqrt( -np.log( eps * (2.0 * bandwidth * np.sqrt(np.pi))**dim ) ) * \
                    2.0 * bandwidth
                
            else:
                # Gaussian kernel only.
                raise NotImplementedError
                
            def function_(bandwidth):
                self.bandwidth = bandwidth
                
                # The linear summand is the expectation of the density estimate.
                mean, std = self.integrate(np.vectorize(lambda x : x))
                linear_term = -2.0 * mean
                
                # The quadratic summand is the sum of the kernel cross-correlations.
                radius = radius_func(bandwidth)
                ind, dist = self.tree_.query_radius(self.tree_.data, radius, return_distance=True)
                squared_term = []
                for index in range(n_samples):
                    squared_term.append(math.fsum(correlation_func(dist[index], bandwidth)))
                squared_term = math.fsum(squared_term) / n_samples**2
                
                # Naive calculation.
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
    Class for evaluating functional based on Kozachenko-Leonenko estimator.
    """

    def __init__(self, *args, k_neighbours: int=5, tree_algorithm: str='ball_tree',
                 tree_params: dict={'leaf_size': 40, 'metric': 'euclidean'}, n_jobs: int=1):
        """
        Initialization.
        
        Parameters
        ----------
        k_neighbours : int
            The number of nearest neighbors used to estimate the density.
        tree_algorithm : str
            Metric tree used for density estimation.
        tree_params : dict
            Metric tree parameters.
        n_jobs : int
            Number of jobs.
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
        Build a kNN density estimate.
        
        Parameters
        ----------
        X : array_like
            I.i.d. samples.
        y : array_like
            Target data (ignored).
        sample_weight : array_like
            Samples weights (ignored).
        fit_weights : bool
            Do the weights selection.
        verbose : int
            Output verbosity.
        """

        if len(X.shape) != 2 or X.shape[0] < self.k_neighbours:
            raise TypeError("X must be of shape (?, >= k_neigbours)")
            
        self.data = X
        
        if self.tree_algorithm == 'ball_tree':
            self.tree_ = BallTree(X, **self.tree_params)
        else:
            raise NotImplementedError
            
        # Select the weights.
        if fit_weights:
            self.set_optimal_weights(verbose=verbose)
        
        
    def get_loo_densities(self, outliers_atol: float=0.0, verbose: int=0):
        """
        Obtaining the density at the points on which the fitting was performed.
        The leave-one-out method is applied.
        
        Parameters
        ----------
        outliers_atol : float
            Absolute tolerance for the density value,
            below which points are discarded as outliers.
        verbose : int
            Output verbosity.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        # Getting `_k_neighbours` nearest neighbors.
        distances, indexes = self.tree_.query(self.tree_.data, self.k_neighbours + 1, return_distance=True)
        distances = distances[:,1:]
        
        # Density values.
        unit_ball_volume = ball_volume(dim)
        
        #psi = np.array([sum(1/n for n in range(1, k - 1))] for k in range(self.k_neighbours)) - np.euler_gamma
        psi = np.zeros(self.k_neighbours)
        psi[0] = -np.euler_gamma
        for index in range(1, self.k_neighbours):
            psi[index] = psi[index - 1] + 1 /  index
            
        densities = np.exp(psi) / (unit_ball_volume * np.power(distances, dim))
        
        # Removing statistical outliers.
        #densities = densities[densities > outliers_atol]
        #n_samples = len(densities)
        
        # If there are fewer than two points left, return None.
        #if n_samples <= 1:
        #    return None
        
        # Normalization.
        densities /= (n_samples - 1)
        
        return densities
    

    def set_optimal_weights(self, rcond: float=1e-6, zero_constraints: bool=True,
                            verbose: int=0):
        """
        Otimal weights selection
        
        Parameters
        ----------
        rcond: float
            Cut-off ratio for small singular values in least squares method.
        zero_constraints: bool
            Add constraints, zeroing some of the weights.
        verbose : int
            Output verbosity.
        """
        
        n_samples, dim = self.tree_.data.shape
        
        if dim <= 4:
            # If the number of utilized neighbours is small, the weights are trivial.
            self.weights = np.zeros(self.k_neighbours)
            self.weights[0] = 1.0
            
        else:
            # Build a linear constraint.
            constraints = []

            # Constraint: the sum equals one.
            constraints.append(np.ones(self.k_neighbours) / self.k_neighbours)

            # Consraint: gamma function.
            n_gamma_constraints = dim // 4
            for k in range(1, n_gamma_constraints + 1):
                constraints.append(
                    #np.array([math.gamma(j + 2*k / dim) / math.gamma(j) for j in range(1, self.k_neighbours + 1)])
                    np.exp(loggamma(np.arange(1, self.k_neighbours + 1) + 2 * k / dim) - \
                           loggamma(np.arange(1, self.k_neighbours + 1)))
                )
                constraints[-1] /= np.linalg.norm(constraints[-1])
                
            # Constraint: zero out some elements.
            if zero_constraints:
                nonzero = set(i * self.k_neighbours // dim - 1 for i in range(1, dim + 1))
                for j in range(self.k_neighbours):
                    if not j in nonzero:
                        constraint = np.zeros(self.k_neighbours)
                        constraint[j] = 1.0
                        constraints.append(constraint)
                    
            constraints = np.vstack(constraints)
            
            # Right hand side.
            rhs = np.zeros(constraints.shape[0])
            rhs[0] = 1.0 / self.k_neighbours

            self.weights = np.linalg.lstsq(constraints, rhs, rcond=rcond)[0]
            #self.weights /= np.sum(self.weights)
        
        return self.weights