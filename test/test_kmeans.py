# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.utils import make_clusters


def test_kmeans_errors():
    """
    Unit test for common failure cases.
    """
    # Raise an error for k=0
    with pytest.raises(ValueError):
        km = KMeans(k=0)

    # Raise an error for max_iter < 1
    with pytest.raises(ValueError):
        km = KMeans(k=4, max_iter=0)
    
    km = KMeans(k=4)

    # Raise an error when done out of order
    with pytest.raises(RuntimeError):
        km.predict(np.array([[1], [2], [3]]))

    # Raise an error for k < data points
    with pytest.raises(ValueError):
        km.fit(np.array([[1], [2], [3]]))

    # Raise an error for no feature dimensions
    with pytest.raises(ValueError):
        km.fit(np.array([[], [], []]))

    # Fit to 2D data
    clusters, labels = make_clusters(k=4, scale=1)
    km.fit(clusters)

    # Raise an error for wrong number of prediction data features
    with pytest.raises(ValueError):
        km.predict(np.array([[1], [2], [3]]))


def test_kmeans():
    """
    Test kmeans runs on various cases.
    """
    km = KMeans(k=4)

    # Test on 1D data, with low SD and very separated clusters
    clusters, labels = make_clusters(k=4, scale=0.1, m=1, bounds=(-100, 100))
    km.fit(clusters)
    predictions = km.predict(clusters)
    mapping = {}
    for i in range(labels.shape[0]):
        if labels[i] not in mapping:
            mapping[labels[i]] = predictions[i]
        else:
            assert mapping[labels[i]] == predictions[i]
    assert km.get_centroids().shape == (4, 1)
    
    # Test on 2D data
    clusters, labels = make_clusters(k=4, scale=0.1, m=2, bounds=(-100, 100))
    km.fit(clusters)
    predictions = km.predict(clusters)
    mapping = {}
    for i in range(labels.shape[0]):
        if labels[i] not in mapping:
            mapping[labels[i]] = predictions[i]
        else:
            assert mapping[labels[i]] == predictions[i]
    assert km.get_centroids().shape == (4, 2)

    # Test on 10D data
    clusters, labels = make_clusters(k=4, scale=0.1, m=10, bounds=(-100, 100))
    km.fit(clusters)
    predictions = km.predict(clusters)
    mapping = {}
    for i in range(labels.shape[0]):
        if labels[i] not in mapping:
            mapping[labels[i]] = predictions[i]
        else:
            assert mapping[labels[i]] == predictions[i]
    assert km.get_centroids().shape == (4, 10)
