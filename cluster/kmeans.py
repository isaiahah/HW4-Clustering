import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Check that k and max_iter are positive
        if k < 1:
            raise ValueError("k should be > 0")
        if max_iter < 1:
            raise ValueError("max_iter should be > 0")
        
        # Store arguments as attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        # Initialize cluster centers as none
        self.cluster_means = None

    def _init_cluster_means(self, mat: np.ndarray) -> np.ndarray:
        """
        Initialize means by randomly choosing k points.
        """
        # Choose random points
        rng = np.random.default_rng(42)
        mean_indices = rng.choice(mat.shape[0], self.k, replace=False)
        self.cluster_means = mat[mean_indices, :]

        # Other option: assign points to clusters randomly, take means
        # assignment = np.random.randint(0, self.k - 1, size=(mat.shape[0], ))
        # self._set_cluster_means(mat, assignment)

    def _init_cluster_kmeanspp(self, mat: np.ndarray) -> np.ndarray:
        """
        Initialize clusters via kmeans++
        """
        # Choose the first random point uniformly
        rng = np.random.default_rng(42)
        first_choice = rng.choice(mat.shape[0], 1)
        self.cluster_means = mat[first_choice, :]

        # Choose points with probability proportional to closest nearest point
        while self.cluster_means.shape[0] < self.k:
            dist = cdist(self.cluster_means, mat)
            prob = np.min(dist, axis=0) ** 2
            prob = prob / np.sum(prob)
            next_choice = rng.choice(mat.shape[0], 1, p=prob)
            self.cluster_means = np.vstack((self.cluster_means, mat[next_choice, :]))
        
        return self.cluster_means

    
    def _set_cluster_means(self, mat: np.ndarray, assignment: np.ndarray):
        """
        Set the cluster means based on the data and assignments.
        """
        self.cluster_means = np.zeros((self.k, mat.shape[1]))
        for mean_index in range(self.k):
            self.cluster_means[mean_index] = np.mean(mat[assignment == mean_index, :], axis=0)

    def _assign_points(self, mat: np.ndarray) -> np.ndarray:
        """
        Return the assignment of data points to this object's cluster means.
        """
        if self.cluster_means is None:
            raise RuntimeError("Attempting to assign points to clusters before cluster means were calculated")

        # Compute distance from each cluster mean to each data point
        dist = cdist(self.cluster_means, mat)
        # Find the closest cluster for each column (for each data point)
        return np.argmin(dist, axis=0)
    
    def _assignment_error(self, mat: np.ndarray, assignment: np.ndarray) -> float:
        """
        Return the total distance of points from the cluster means.
        """
        if self.cluster_means is None:
            raise RuntimeError("Attempting to calculate error before cluster means were calculated")
        
        # Compute distance from each cluster mean to each data point
        dist = cdist(self.cluster_means, mat)
        # Sum the distance of each point from its assigned center
        error = 0
        for i in range(mat.shape[0]):
            error += dist[assignment[i], i] ** 2
        return error

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Raise error if fewer data points than clusters
        if mat.shape[0] < self.k:
            raise ValueError(f"Data should contain at least {self.k} values")
        
        # Raise error if no feature dimensions
        if mat.shape[1] == 0:
            raise ValueError(f"Data should contain at least one feature")

        # Initialze cluster means by randomly selecting k data points
        self._init_cluster_kmeanspp(mat)

        # Initialize cluster assignments and calculate the initial error
        assignment = self._assign_points(mat)
        self.error = self._assignment_error(mat, assignment)

        # k-means loop: assign points to closest cluster mean, then recalculate means
        iter = 0
        while iter < self.max_iter:
            # Assign points to closest cluster mean
            assignment = self._assign_points(mat)
            # Recalculate means
            self._set_cluster_means(mat, assignment)
            # Calculate new error and terminate difference is if below tolerance
            error = self._assignment_error(mat, assignment)
            if self.error - error < self.tol:
                break
            self.error = error
            iter += 1

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.cluster_means is None:
            raise RuntimeError("Attempting to calculate clusters before cluster means were calculated")
        if mat.shape[1] != self.cluster_means.shape[1]:
            raise ValueError("Input data is not of same dimension as training data")
        
        return self._assign_points(mat)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.cluster_means
