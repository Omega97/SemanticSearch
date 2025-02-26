from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        Default model: 'all-MiniLM-L6-v2' (small & efficient)
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Converts a list of text inputs into vector embeddings.

        Args:
            texts (list of str): List of text inputs.

        Returns:
            np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim]).
        """
        return np.array(self.model.encode(texts, convert_to_numpy=True))
