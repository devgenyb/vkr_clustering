from sklearn.feature_extraction.text import TfidfVectorizer as tdidf
from .IVectorizer import InterfaceVectorizer



class TfidfVectorizer(InterfaceVectorizer):
    def __init__(self):
        self.model = tdidf(max_df=1)

    def vectorize(self, data):
        return tdidf(min_df=1).fit_transform(data).toarray()
