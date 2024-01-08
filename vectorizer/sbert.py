from enum import Enum
from .IVectorizer import InterfaceVectorizer
from sentence_transformers import SentenceTransformer


class SbertModels(Enum):
    DEEPPAVLOV = 'DeepPavlov/rubert-base-cased-sentence'
    PARAPHRASE = 'paraphrase-multilingual-MiniLM-L12-v2'
    DISTILUSE = 'distiluse-base-multilingual-cased'


class SbertVectoriver(InterfaceVectorizer):
    def __init__(self, sbert_model: SbertModels):
        self.model = SentenceTransformer(sbert_model.value)

    def vectorize(self, data):
        return self.model.encode(data)

