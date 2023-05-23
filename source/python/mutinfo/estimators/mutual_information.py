import math
import numpy as np
from scipy.linalg import block_diag
from collections  import Counter
from .entropy import EntropyEstimator


class MutualInfoEstimator:
    """
    Mutual information estimator.
    """

    def __init__(self, X_is_discrete: bool=False, Y_is_discrete: bool=False,
                 entropy_estimator_params: dict={'method': 'KL', 'functional_params': None}):
        """
        Initialization.

        Parameters
        ----------
        X_is_discrete : bool
            X is a discrete random variable.
        Y_is_discrete : bool
            Y is a discrete random variable.
        entropy_estimator_params : dict
            Entropy estimator parameters.
        """

        self._X_is_discrete = X_is_discrete
        self._Y_is_discrete = Y_is_discrete
        self.entropy_estimator_params = entropy_estimator_params

        self._X_entropy_estimator = None
        self._Y_entropy_estimator = None
        self._X_Y_entropy_estimator = None


    def fit(self, X, Y, verbose: int=0):
        """
        Fit parameters of the estimator.

        Parameters
        ----------
        X : Iterable
            I.i.d. samples from X.
        Y : Iterable
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same length")

        if not self._X_is_discrete and not self._Y_is_discrete:
            if self.entropy_estimator_params['method'] == 'KDE':
                if verbose >= 1:
                    print("Fitting the estimator for (X,Y)")

                self._X_Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_Y_entropy_estimator.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Fitting the estimator for X")

                self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_entropy_estimator.fit(X, fit_bandwidth=False, verbose=verbose)

                if verbose >= 1:
                    print("Fitting the estimator for Y")

                self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._Y_entropy_estimator.fit(Y, fit_bandwidth=False, verbose=verbose)

                # Bandwidth value is shared among estimators.
                bandwidth = self._X_Y_entropy_estimator._functional.bandwidth
                self._X_entropy_estimator._functional.bandwidth = bandwidth
                self._Y_entropy_estimator._functional.bandwidth = bandwidth
                
            else:
                if verbose >= 1:
                    print("Fitting the estimator for (X,Y)")

                self._X_Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_Y_entropy_estimator.fit(np.concatenate([X, Y], axis=1), verbose=verbose)

                if verbose >= 1:
                    print("Fitting the estimator for X")

                self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._X_entropy_estimator.fit(X, verbose=verbose)

                if verbose >= 1:
                    print("Fitting the estimator for Y")

                self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
                self._Y_entropy_estimator.fit(Y, verbose=verbose)
                
        elif self._X_is_discrete and not self._Y_is_discrete:
            if verbose >= 1:
                print("Fitting the estimator for Y")

            self._Y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            self._Y_entropy_estimator.fit(Y, verbose=verbose)

        elif not self._X_is_discrete and self._Y_is_discrete:
            if verbose >= 1:
                print("Fitting the estimator for X")

            self._X_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            self._X_entropy_estimator.fit(X, verbose=verbose)

        else:
            # No fitting is required when both variables are discrete.
            pass


    def estimate(self, X, Y, verbose: int=0):
        """
        Mutual information estimation.

        Parameters
        ----------
        X : Iterable
            I.i.d. samples from X.
        Y : Iterable
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same length")

        if not self._X_is_discrete and not self._Y_is_discrete:
            return self._estimate_cont_cont(X, Y, verbose=verbose)

        elif self._X_is_discrete and not self._Y_is_discrete:
            return self._estimate_cont_disc(Y, X, self._Y_entropy_estimator, verbose=verbose)

        elif not self._X_is_discrete and self._Y_is_discrete:
            return self._estimate_cont_disc(X, Y, self._X_entropy_estimator, verbose=verbose)

        else:
            return self._estimate_disc_disc(X, Y, verbose=verbose)


    def _estimate_cont_cont(self, X, Y, verbose: int=0):
        """
        Mutual information estimation for a pair of absolutely continuous random variables.

        Parameters
        ----------
        X : array_like
            I.i.d. samples from X.
        Y : array_like
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        if verbose >= 1:
            print("Entropy estimation for X")
        H_X, H_X_err = self._X_entropy_estimator.estimate(X, verbose=verbose)

        if verbose >= 1:
            print("Entropy estimation for Y")
        H_Y, H_Y_err = self._Y_entropy_estimator.estimate(Y, verbose=verbose)

        if verbose >= 1:
            print("Entropy estimation for (X,Y)")
        H_X_Y, H_X_Y_err = self._X_Y_entropy_estimator.estimate(np.concatenate([X, Y], axis=1), verbose=verbose)

        return (H_X + H_Y - H_X_Y, H_X_err + H_Y_err + H_X_Y_err)


    def _estimate_cont_disc(self, X, Y, X_entropy_estimator: EntropyEstimator,
                            verbose: int=0):
        """
        Mutual information estimation for an absolutely continuous X and discrete Y.
        
        Parameters
        ----------
        X : array_like
            I.i.d. samples from X.
        Y : Iterable
            I.i.d. samples from Y.
        X_entropy_estimator : EntropyEstimator
            Entropy estimator for X.
        verbose : int
            Output verbosity.
        """

        if verbose >= 1:
            print("Entropy estimation for the absolutely continuous random variable")
        H_X, H_X_err = X_entropy_estimator.estimate(X, verbose=verbose)

        # Empirical frequencies estimation.
        frequencies = Counter(Y)
        for y in frequencies.keys():
            frequencies[y] /= Y.shape[0]

        if verbose >= 2:
            print("Empirical frequencies: ")
            print(frequencies)

        # Conditional entropy estimation.
        H_X_mid_y = dict()
        for y in frequencies.keys():
            X_mid_y = X[Y == y]

            # For every value of Y it is required to refit the estimator.
            if verbose >= 1:
                print("Conditional entropy estimation for the absolutely continuous random variable")
            X_mid_y_entropy_estimator = EntropyEstimator(**self.entropy_estimator_params)
            X_mid_y_entropy_estimator.fit(X_mid_y, verbose=verbose)
            H_X_mid_y[y] = X_mid_y_entropy_estimator.estimate(X_mid_y, verbose=verbose)

        # Final conditional entropy estimate.
        cond_H_X     = math.fsum([frequencies[y] * H_X_mid_y[y][0] for y in frequencies.keys()])
        cond_H_X_err = math.fsum([frequencies[y] * H_X_mid_y[y][1] for y in frequencies.keys()])

        return (H_X - cond_H_X, H_X_err + cond_H_X_err)


    def _estimate_disc_disc(self, X, Y, verbose: int=0):
        """
        Mutual information estimation for a pair of discrete random variables.

        Parameters
        ----------
        X : array_like
            I.i.d. samples from X.
        Y : array_like
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        H_X = 0.0
        H_Y = 0.0
        H_X_Y = 0.0

        frequencies_X = Counter(X)
        for x in frequencies_X.keys():
            frequencies_X[x] /= X.shape[0]
            H_X -= frequencies_X[x] * np.log(frequencies_X[x])

        frequencies_Y = Counter(Y)
        for y in frequencies_Y.keys():
            frequencies_Y[y] /= Y.shape[0]
            H_Y -= frequencies_Y[y] * np.log(frequencies_Y[y])

        frequencies_X_Y = Counter(np.concatenate([X, Y], axis=1))
        for x_y in frequencies_X_Y.keys():
            frequencies_X_Y[x_y] /= X.shape[0]
            H_X_Y -= frequencies_X_Y[x_y] * np.log(frequencies_X_Y[x_y])

        if verbose >= 2:
            print("Empirical frequencies of X: ")
            print(frequencies_X)

            print("Empirical frequencies of Y: ")
            print(frequencies_Y)

            print("Empirical frequencies of (X, Y): ")
            print(frequencies_X_Y)

        return (H_X + H_Y - H_X_Y, 0.0)



class LossyMutualInfoEstimator(MutualInfoEstimator):
    """
    Mutual information estimator, complemented with lossy compressor.
    """

    def __init__(self, X_compressor: callable=None, Y_compressor: callable=None,
                 *args, **kwargs):
        """
        Initialization.

        Parameters
        ----------
        X_compressor : callable
            Callable object that performs compression of X.
        Y_compressor : callable
            Callable object that performs compression of Y.
        """

        super().__init__(*args, **kwargs)
        self._X_compressor = X_compressor
        self._Y_compressor = Y_compressor


    def fit(self, X, Y, verbose: int=0):
        """
        Fit parameters of the estimator.

        Parameters
        ----------
        X : Iterable
            I.i.d. samples from X.
        Y : Iterable
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        X_compressed = X if self._X_compressor is None else self._X_compressor(X)
        Y_compressed = Y if self._Y_compressor is None else self._Y_compressor(Y)
        super().fit(X_compressed, Y_compressed, verbose=verbose)

        
    def estimate(self, X, Y, verbose: int=0):
        """
        Mutual information estimation.

        Parameters
        ----------
        X : Iterable
            I.i.d. samples from X.
        Y : Iterable
            I.i.d. samples from Y.
        verbose : int
            Output verbosity.
        """

        X_compressed = X if self._X_compressor is None else self._X_compressor(X)
        Y_compressed = Y if self._Y_compressor is None else self._Y_compressor(Y)
        return super().estimate(X_compressed, Y_compressed, verbose=verbose)