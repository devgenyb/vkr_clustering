import numpy as np
import random
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time


class myKMeans:
    def __init__(self, n_clusters, iters=300, distance='euclidean'):
        """
        KMeans Class constructor.

        Args:
          n_clusters (int) : Number of clusters used for partitioning.
          iters (int) : Number of iterations until the algorithm stops.

        """
        self.n_clusters = n_clusters
        self.iters = iters
        self.distance = distance
        self.labels = None
        self.centers = None
        self.inertia_ = 0

    def kmeans_plus_plus(self, X, n_clusters):
        """
        My implementation of the KMeans++ initialization method for computing the centroids.

        Args:
            X (ndarray): Dataset samples
            n_clusters (int): Number of clusters

        Returns:
            centroids (ndarray): Initial position of centroids
        """
        # Assign the first centroid to a random sample from the dataset.
        idx = random.randrange(len(X))
        centroids = [X[idx]]

        # For each cluster
        for _ in range(1, n_clusters):
            # squared_distances = np.array(
            #     [min([np.inner(centroid - sample, centroid - sample) for centroid in centroids]) for sample in X])
            # proba = squared_distances / squared_distances.sum()
            #
            # for point, probability in enumerate(proba):
            #     if probability == proba.max():
            #         centroid = point
            #         break

            distances = pairwise_distances(X, np.array(centroids), metric=self.distance)
            min_distances = np.min(distances, axis=1)
            if self.distance == 'euclidean':
                min_distances = min_distances ** 2

            proba = min_distances / min_distances.sum()
            centroid = np.argmax(np.random.multinomial(1, proba))

            centroids.append(X[centroid])

        return np.array(centroids)

    def find_closest_centroids(self, X, centroids):
        """
        Computes the distance to the centroids and assigns the new label to each sample in the dataset.

        Args:
            X (ndarray): Dataset samples
            centroids (ndarray): Number of clusters

        Returns:
            idx (ndarray): Closest centroids for each observation

        """

        # Set K as number of centroids
        K = centroids.shape[0]

        # Initialize the labels array to 0
        label = np.zeros(X.shape[0], dtype=int)

        # For each sample in the dataset
        for sample in range(len(X)):
            distance = []
            # Take every centroid
            for centroid in range(len(centroids)):
                # Compute Euclidean norm between a specific sample and a centroid
                # norm = np.linalg.norm(X[sample] - centroids[centroid])
                # distance.append(norm)
                # Вычисляем евклидово расстояние от текущей точки данных до каждого центроида
                distances = pairwise_distances(X[sample].reshape(1, -1), centroids, metric=self.distance)[0]

                # Находим индекс центроида с минимальным расстоянием и присваиваем метку
                label[sample] = np.argmin(distances)
            # Assign the closest centroid as it's label
            # label[sample] = distance.index(min(distance))

        return label

    def compute_centroids(self, X, idx, K):
        """
        Returns the new centroids by computing the mean of the data points assigned to each centroid.

        Args:
            X (ndarray): Dataset samples
            idx (ndarray): Closest centroids for each observation
            K (int): Number of clusters

        Returns:
            centroids (ndarray): New centroids computed
        """

        # Number of samples and features
        m, n = X.shape

        # Initialize centroids to 0
        centroids = np.zeros((K, n))

        # For each centroid
        for k in range(K):
            # Take all samples assigned to that specific centroid
            points = X[idx == k]
            # Compute their mean
            centroids[k] = np.mean(points, axis=0)

        return centroids

    def compute_inertia(self, X, centroids, labels):
        distances_matrix = pairwise_distances(X, centroids, metric=self.distance)

        # Для каждой точки данных и ее соответствующего кластера
        inertia = 0
        for sample, cluster_label in zip(distances_matrix, labels):
            # Выбираем квадрат расстояния от текущей точки данных до центроида ее кластера
            inertia += sample[cluster_label] ** 2

        self.inertia_ = inertia
        return inertia

    def fit_predict(self, X):
        """
        Args:
            X (ndarray): Dataset samples

        Returns:
            centroids (ndarray):  Computed centroids
            labels (ndarray):     Predicts for each sample in the dataset.
        """

        start_time = time.time()

        # Number of samples and features
        m, n = X.shape

        # Compute initial position of the centroids
        initial_centroids = self.kmeans_plus_plus(X, self.n_clusters)

        centroids = initial_centroids
        labels = np.zeros(m)

        prev_centroids = centroids

        # Run K-Means
        for i in range(self.iters):
            # For each example in X, assign it to the closest centroid
            labels = self.find_closest_centroids(X, centroids)

            # Given the memberships, compute new centroids
            centroids = self.compute_centroids(X, labels, self.n_clusters)

            # Check if centroids stopped changing positions
            if centroids.tolist() == prev_centroids.tolist():
                end_time = time.time()
                print(f'K-Means converged at {i + 1} iterations, from {end_time - start_time} time')
                break
            else:
                prev_centroids = centroids

        self.compute_inertia(X, centroids, labels)
        return centroids, labels


def my_elbow_method(X, max_clusters, distance='euclidean'):
    k_values = range(2, max_clusters)
    inertia_values = []
    silhouette_values = []

    for k in k_values:
        model = myKMeans(n_clusters=k, distance=distance)
        centroids, labels = model.fit_predict(X)
        inertia_values.append(model.inertia_)
        silhouette_values.append(silhouette_score(X, labels))

    ind = 2
    max = silhouette_values[0]
    for i in range(len(silhouette_values)):
        if i > max:
            max = i
            ind = i + 2


    return inertia_values, silhouette_values, ind
