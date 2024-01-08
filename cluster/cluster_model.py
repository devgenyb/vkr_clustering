from enum import Enum
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


class ModelType(Enum):
    KMEANS = 'kmeans'
    DBSCAN = 'dbscan'
    HIERARCHICAL = 'hierarchical'


class ClusterModel:
    def __init__(self, model: ModelType, parameter: int | float):
        if model == ModelType.KMEANS.value[0]:
            self.model = KMeans(n_clusters=parameter, init='k-means++', random_state=42)
        if model == ModelType.DBSCAN.value[0]:
            self.model = DBSCAN(eps=parameter, min_samples=2)
        if model == ModelType.HIERARCHICAL.value[0]:
            self.model = AgglomerativeClustering(n_clusters=parameter, linkage='ward')

    def run_model(self, X):
        return self.model.fit_predict(X)
