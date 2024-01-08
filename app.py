from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import functions
from vectorizer.main import Vectorizer
from vectorizer.dfidf import TfidfVectorizer
from vectorizer.sbert import SbertVectoriver, SbertModels
from enum import Enum
from cluster.cluster_model import ClusterModel, ModelType

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


class VectorizeMethods(Enum):
    DFIDF = 'df-idf',
    SBERT = 'sbert'


@app.route('/visualizer', methods=['POST'])
def visualizer():
    data = json.loads(request.form['data'])
    file = request.files['file']
    df = pd.read_excel(file)
    df = functions.exel_preprocessing(df)
    if data['vectorMethod']['code'] == VectorizeMethods.DFIDF.value[0]:
        vectorizer = Vectorizer(TfidfVectorizer())
    elif data['vectorMethod']['code'] == VectorizeMethods.SBERT.value:
        vectorizer = Vectorizer(SbertVectoriver(sbert_model=SbertModels.DEEPPAVLOV))
    vectors = vectorizer.vectorize(df['Фраза'])
    elbow, silluete = functions.kmeans_metrics(vectors, data['maxclusters'])
    clusters_forms_image = functions.vectors_image(vectors)
    elbow_image = functions.plot_image(elbow, 'график локтя', 'n кластеров', 'счет', x_from=2, image_name='elbow_image')
    siluete_image = functions.plot_image(silluete, 'график силуэта', 'n кластеров', 'счет', x_from=2, image_name='sil_image')
    dendogram_image = functions.dendogram_image(vectors, 'dendogram')
    three_max_values_with_indices = sorted(enumerate(silluete), key=lambda x: x[1], reverse=True)[:3]
    three_max_indices = [index+2 for index, value in three_max_values_with_indices]
    return jsonify({
        'clusters_forms': clusters_forms_image,
        'elbow': elbow_image,
        'silluete': siluete_image,
        'dendogram': dendogram_image,
        'n_clusters': three_max_indices,
    })


@app.route('/clustering', methods=['POST'])
def clustering():
    data = json.loads(request.form['data'])
    file = request.files['file']
    df = pd.read_excel(file)
    df = functions.exel_preprocessing(df)
    centers = False
    if data['vectorMethod']['code'] == VectorizeMethods.DFIDF.value[0]:
        vectorizer = Vectorizer(TfidfVectorizer())
    elif data['vectorMethod']['code'] == VectorizeMethods.SBERT.value:
        vectorizer = Vectorizer(SbertVectoriver(sbert_model=SbertModels.DEEPPAVLOV))
    vectors = vectorizer.vectorize(df['Фраза'])
    if data['algMethod']['code'] == 'kmeans':
        model = ClusterModel(model=ModelType.KMEANS.value[0], parameter=data['nclasters'])
        centers = True
    if data['algMethod']['code'] == 'dbscan':
        model = ClusterModel(model=ModelType.DBSCAN.value[0], parameter=data['dbscaneps'])
    if data['algMethod']['code'] == 'Aglomirative':
        model = ClusterModel(model=ModelType.HIERARCHICAL.value[0], parameter=data['nclasters'])
    labels = model.run_model(vectors)
    if centers:
        centers = model.model.cluster_centers_
    else:
        centers = None
    df['claster'] = labels
    df = functions.post_processing(df)
    diagram = functions.create_cluster_diagram_image(vectors, labels, centers, 'clusters_ready_diagram')
    ready_file = functions.create_exel_from_df(df, 'ready_table')
    return jsonify({
        'diagram': diagram,
        'file': ready_file
    })







if __name__ == '__main__':
    app.run(debug=True)
