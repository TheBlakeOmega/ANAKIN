import pickle
from gensim.models import Word2Vec


class EmbeddingModel:

    def __init__(self):
        self.stringEmbeddingModel = None

    def loadModel(self, load_path):
        with open(load_path + '/embeddingModel.pkl', 'rb') as f:
            self.stringEmbeddingModel = pickle.load(f)

    def saveModel(self, save_path):
        with open(save_path + '/embeddingModel.pkl', 'wb') as f:
            pickle.dump(self.stringEmbeddingModel, f)

    def train(self, extracted_strings, vector_size):
        self.stringEmbeddingModel = Word2Vec(extracted_strings,
                                             vector_size=vector_size, window=5, min_count=0, workers=4, seed=42)

    def getEmbeddedString(self, input_string):
        return self.stringEmbeddingModel.wv[input_string]

    def getEmbeddedStringLength(self):
        return self.stringEmbeddingModel.wv.vector_size
