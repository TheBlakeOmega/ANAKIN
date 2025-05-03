import pickle
from gensim.models import Word2Vec


class EmbeddingModel:
    """
    A class to manage the training, saving, loading, and usage of a Word2Vec embedding model.
    """

    def __init__(self):
        """
        Initializes the EmbeddingModel object.
        """
        self.stringEmbeddingModel = None

    def loadModel(self, load_path):
        """
        Loads the embedding model from a specified path.

        Args:
            load_path (str): Path to the directory containing the embedding model file.

        Returns:
            None
        """
        with open(load_path + '/embeddingModel.pkl', 'rb') as f:
            self.stringEmbeddingModel = pickle.load(f)

    def saveModel(self, save_path):
        """
        Saves the embedding model to a specified path.

        Args:
            save_path (str): Path to the directory where the embedding model file will be saved.

        Returns:
            None
        """
        with open(save_path + '/embeddingModel.pkl', 'wb') as f:
            pickle.dump(self.stringEmbeddingModel, f)

    def train(self, extracted_strings, vector_size):
        """
        Trains the embedding model using the provided strings.

        Args:
            extracted_strings (list of list of str): List of tokenized strings to train the model on.
            vector_size (int): Dimensionality of the embedding vectors.

        Returns:
            None
        """
        self.stringEmbeddingModel = Word2Vec(extracted_strings,
                                             vector_size=vector_size, window=5, min_count=0, workers=4, seed=42)

    def getEmbeddedString(self, input_string):
        """
        Retrieves the embedding vector for a given string.

        Args:
            input_string (str): The string to retrieve the embedding for.

        Returns:
            numpy.ndarray: The embedding vector for the input string.
        """
        return self.stringEmbeddingModel.wv[input_string]

    def getEmbeddedStringLength(self):
        """
        Retrieves the dimensionality of the embedding vectors.

        Returns:
            int: The size of the embedding vectors.
        """
        return self.stringEmbeddingModel.wv.vector_size
