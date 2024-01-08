from abc import ABC, abstractmethod

class InterfaceVectorizer(ABC):

    @abstractmethod
    def vectorize(self, data):
        pass

