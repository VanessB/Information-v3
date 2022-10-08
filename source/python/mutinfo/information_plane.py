import numpy as np
from .estimators.mutual_information import LossyMutualInfoEstimator

class InformationPlaneAnalyser:
    """
    Анализатор обучения модели на информационной плоскости.
    """

    def __init__(self, X_compressor=None, Y_compressor=None, Y_is_discrete=False, n_jobs=1):
        """
        X_compressor  - компрессор для входов модели.
        Y_compressor  - компрессор для выходов модели.
        Y_is_discrete - является ли выход модели дискретным.
        """

        self.X_compressor_ = X_compressor
        self.Y_compressor_ = Y_compressor
        self.Y_is_discrete_ = Y_is_discrete
        self.n_jobs_ = n_jobs

        # Списки для хранения взаимных информаций.
        self.X_L_MI = []
        self.L_Y_MI = []

    def step(self, X, Y, Ls, L_compressors=None):
        """
        Выполнить шаг алгоритма.
        X - входные данные.
        Y - данные меток.
        L - данные, для которых требуется вычислить I(X,L) и I(L,Y).
        """

        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == Ls.shape[0]
        n_outputs = Ls.shape[1]

        if L_compressors is None:
            L_compressors = [None] * n_outputs

        X_L_mi = []
        L_Y_mi = []
        for output_index in range(n_outputs):
            L = L[:,output_index]
            L_compressor = L_compressors[output_index]

            X_L_mi_estimator = LossyMutualInfoEstimator(X_compressor, L_compressor, n_jobs=self.n_jobs_)
            X_L_mi_estimator.fit(X, L)
            X_L_mi.append(X_L_mi_estimator.predict(X, L))

            L_Y_mi_estimator = LossyMutualInfoEstimator(L_compressor, Y_compressor, Y_is_discrete=self.Y_is_discrete, n_jobs=self.n_jobs_)
            L_Y_mi_estimator.fit(L, Y)
            L_Y_mi.append(L_Y_mi_estimator.predict(L, Y))

        self.X_L_MI.append(X_L_mi)
        self.L_Y_MI.append(L_Y_mi)