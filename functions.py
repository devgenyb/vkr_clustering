import pandas
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.decomposition import PCA
import pandas as pd
from pymorphy3 import MorphAnalyzer
from scipy.cluster.hierarchy import dendrogram, linkage


def find_stop_words(list_of_stop_words):
    def closure(keyword):
        keyword = keyword.split()
        for word in list_of_stop_words['stop_slova']:
            if word in keyword:
                return ' '.join([w for w in keyword if w != word])

    return closure


def kmeans_metrics(X, max_clusters):
    k_values = range(2, max_clusters)
    inertia_values = []
    silhouette_values = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        inertia_values.append(model.inertia_)
        silhouette_values.append(silhouette_score(X, labels))

    return inertia_values, silhouette_values


def jacar_index(df):
    if max(df['Кластер']) < 2:
        return 0
    jacar_metric_clusters = []
    cluster_count_manual = df.max()['Разметка'] + 1

    for i in range(0, cluster_count_manual):
        current = df[df['Разметка'] == i]
        count = [0] * cluster_count_manual
        for j in range(current.shape[0]):
            row = current.iloc[[j]]
            cluster_index = int(row['Кластер'])
            if cluster_index == -1:
                continue
            count[cluster_index] = count[cluster_index] + 1

        current_max = count[0]
        c_index = 0
        for count_index in range(len(count)):
            if count[count_index] > current_max:
                current_max = count[count_index]
                c_index = count_index

        manual_cluster_index = i
        if c_index != manual_cluster_index:
            current['Кластер'] = current['Кластер'].replace([manual_cluster_index], -1)
            current['Кластер'] = current['Кластер'].replace([c_index], manual_cluster_index)
        jacar_metric_clusters.append(jaccard_score(current['Разметка'], current['Кластер'], average='micro'))

    return sum(jacar_metric_clusters) / len(jacar_metric_clusters)


def clustering__diagram(X, labels, title='title', xlabel='x', ylabel='y', image_name='clasters_form'):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plt.savefig('./static/generated/' + image_name + '.png')


def vectors_image(vectors, image_name='vectors_image'):
    pca = PCA(n_components=2)
    name_of_vector_array = vectors
    df2d = pd.DataFrame(pca.fit_transform(name_of_vector_array), columns=list('xy'))
    df2d.plot(kind='scatter', x='x', y='y')
    plt.title('Форма векторов')
    path = './static/generated/' + image_name + '.png'
    plt.savefig(path)
    plt.clf()
    return path[1:]


def plot_image(points, label, x_label, y_label, x_from=2, image_name='vectors_image'):
    x = range(x_from, x_from + len(points))
    plt.title(label)
    plt.scatter(x, points)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    path = './static/generated/' + image_name + '.png'
    plt.savefig(path)
    plt.clf()
    return path[1:]


def dendogram_image(vectors: np.ndarray, image_name: str ='dendogram_image') -> str:
    linkage_matrix = linkage(vectors, 'ward')
    dendrogram(linkage_matrix, labels=None)
    plt.title('Иерархическая дендрограмма')
    plt.xlabel('индексы')
    plt.ylabel('дистанция')
    path = './static/generated/' + image_name + '.png'
    plt.savefig(path)
    plt.clf()
    return path[1:]


def exel_preprocessing(df: pandas.DataFrame) -> pandas.DataFrame:
    list_of_stop_words = pd.read_excel('./helps_files/spisok_stop_slov.xlsx', names=['stop_slova'])
    df['Фраза_по_словам'] = df['Фраза'].str.split()
    m = MorphAnalyzer()
    df['Леммы'] = [' '.join([m.parse(word)[0].normal_form for word in x.split()]) for x in df['Фраза']]
    df['Леммы'].apply(find_stop_words(list_of_stop_words))
    return df


def post_processing(df: pandas.DataFrame) -> pandas.DataFrame:
    return df


def create_exel_from_df(df: pd.DataFrame, file_name: str = 'result') -> str:
    path = './static/generated/' + file_name + '.xlsx'
    write_kernel = pd.ExcelWriter(path, engine='xlsxwriter')
    df.to_excel(write_kernel)
    write_kernel.close()
    return path[1:]


def create_cluster_diagram_image(vectors: np.ndarray, labels: np.ndarray, centers: np.ndarray | None = None,
                                    image_name: str = 'clusters_diagram'
                                 ) -> str:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(vectors)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    # if centers is not None:
    #     plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, color='red', label='Centroids')
    plt.title('Диаграмма кластеров')
    plt.legend()
    path = './static/generated/' + image_name + '.png'
    plt.savefig(path)
    plt.clf()
    return path[1:]

