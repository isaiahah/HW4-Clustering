import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimension mismatch between X and y")
        
        # Compute pairwise distances between all points in X
        pairwise_dist = cdist(X, X)
        # Compute "a" values (mean intra cluster distance)
        # a(i) = (sum_{j in C_I, i != j} d(i, j)) / (|C_I| - 1)
        A = np.zeros(y.shape)
        for i in range(y.shape[0]):
            if np.sum(y == y[i]) > 1: # If only point in the cluster, keep as 0
                others = 0
                for j in range(y.shape[0]):
                    if y[i] == y[j] and i != j:
                        A[i] += pairwise_dist[i, j]
                        others += 1
                A[i] /= others
        # Compute "b" values (mean distance to points in nearest neighbouring cluster)
        # b(i) = min_{J != I} (sum_{j in C_J} d(i, j)) / (|C_J|)
        B = np.zeros(y.shape)
        for i in range(y.shape[0]):
            B[i] = min(np.mean(pairwise_dist[i, y == j]) for j in set(y) if j != y[i])
        # Compute silhouette values
        S = np.zeros(y.shape)
        for i in range(y.shape[0]):
            if np.sum(y == y[i]) > 1: # If only point in cluster, keep as 0
                S[i] = (B[i] - A[i]) / (max(A[i], B[i]))
        return S

