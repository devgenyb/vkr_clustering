from .IVectorizer import InterfaceVectorizer

class Vectorizer:
    def __init__(self, vectorizer: InterfaceVectorizer):
        self.vectorizer = vectorizer

    def vectorize(self, data):
        return self.vectorizer.vectorize(data)
