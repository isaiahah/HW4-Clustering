# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.utils import make_clusters
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_samples

def test_silhouette():
    """
    Test silhouette scores on simple samples
    """
    km = KMeans(k=4)

    # Test on 1D data, with low SD and very separated clusters
    clusters, labels = make_clusters(k=4, scale=1, m=1, bounds=(-10, 10))
    km.fit(clusters)
    predictions = km.predict(clusters)
    error = np.abs(Silhouette().score(clusters, predictions) - silhouette_samples(clusters, predictions))
    assert np.all(error < 0.01)
    
    # Test on 2D data
    clusters, labels = make_clusters(k=4, scale=1, m=2, bounds=(-10, 10))
    km.fit(clusters)
    predictions = km.predict(clusters)
    error = np.abs(Silhouette().score(clusters, predictions) - silhouette_samples(clusters, predictions))
    assert np.all(error < 0.01)

    # Test on 10D data
    clusters, labels = make_clusters(k=4, scale=1, m=10, bounds=(-10, 10))
    km.fit(clusters)
    predictions = km.predict(clusters)
    error = np.abs(Silhouette().score(clusters, predictions) - silhouette_samples(clusters, predictions))
    assert np.all(error < 0.01)

