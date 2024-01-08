import pandas as pd
import functions
from pymorphy3 import MorphAnalyzer
from vectorizer.main import Vectorizer
from vectorizer.sbert import SbertModels, SbertVectoriver
from vectorizer.dfidf import TfidfVectorizer
from sklearn.cluster import KMeans, dbscan, affinity_propagation, mean_shift
from sklearn.metrics import silhouette_score
from nltk.cluster import KMeansClusterer
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cluster.kmeans import myKMeans, my_elbow_method
from functions import jacar_index
from sklearn.cluster import AgglomerativeClustering


# загрузка доп файлов
list_of_stop_words = pd.read_excel('./helps_files/spisok_stop_slov.xlsx', names=['stop_slova'])
list_of_cities = pd.read_excel('./helps_files/spisok_gorodov.xlsx', names=['goroda'])
list_of_pretext = pd.read_excel('./helps_files/spisok_predlogov.xlsx', names=['predlogi'])
list_of_questions = pd.read_excel('./helps_files/spisok_vopros_slov.xlsx', names=['vopros'])
list_of_commerce_words = pd.read_excel('./helps_files/spisok_kommercheskih_slov.xlsx', names=['komm'])


# keywords dataframe
data = pd.read_excel('keywords_test.xlsx')

data['Фраза_по_словам'] = data['Фраза'].str.split()

m = MorphAnalyzer()
data['Леммы'] = [' '.join([m.parse(word)[0].normal_form for word in x.split()]) for x in data['Фраза']]

data['Леммы'].apply(functions.find_stop_words(list_of_stop_words))

vectorized_lemmas_tfidf = Vectorizer(TfidfVectorizer()).vectorize(data['Леммы'])
vectorized_lemmas_sbert = Vectorizer(SbertVectoriver(SbertModels.PARAPHRASE)).vectorize(data['Леммы'])

# Set PCA to 2D [sklearn]
pca = PCA(n_components=2)

# Name of Vector Array (Numpy)
name_of_vector_array = vectorized_lemmas_sbert

# New D2 Dataframe (PCA)
df2d = pd.DataFrame(pca.fit_transform(name_of_vector_array), columns=list('xy'))

# Plot Data Visualization (Matplotlib)
df2d.plot(kind='scatter', x='x', y='y')
plt.show()

kmeans = KMeans(n_clusters=5)
sklearn_labels = kmeans.fit_predict(vectorized_lemmas_sbert)
kmeans_sk = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
sk_labels = kmeans_sk.fit_predict(vectorized_lemmas_tfidf)
data['Кластер'] = sk_labels
js = jacar_index(sklearn_labels)
print(js)


distances = ['sqeuclidean', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
'braycurtis', 'chebyshev', 'correlation',
             ]
# my_kmeans = myKMeans(5, distance='correlation')
# _, labels = my_kmeans.fit_predict(vectorized_lemmas_sbert)
score_table = pd.DataFrame([], columns= ['метрика'] + distances)
score_table['метрика'] = ['Жакар+сберт', 'силуэт+сберт', 'Жакар+dfidf', 'силуэт+dfidf']
data['Кластер'] = sk_labels


