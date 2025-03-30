"""
This module contains the EmbeddingModel class, which is used to convert
text inputs into vector embeddings. The class uses the SentenceTransformer
library to encode the text inputs.

Useful models:
- all-MiniLM-L6-v2 (small & efficient)
- paraphrase-MiniLM-L6-v2 (small & efficient)
- stsb-distilroberta-base-v2 (small)
- paraphrase-mpnet-base-v2 (medium)
- paraphrase-TinyBERT-L6-v2 (medium)
- paraphrase-distilroberta-base-v1 (medium)
- paraphrase-TinyBERT-L6-v2 (medium)
- paraphrase-multilingual-mpnet-base-v2 (multilingual)
- paraphrase-xlm-r-multilingual-v1 (multilingual)
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    The EmbeddingModel class is used to convert text inputs into vector embeddings.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True) -> np.array:
        """
        Converts a list of text inputs into vector embeddings.

        Args:
            texts (list of str): List of text inputs.
            normalize_embeddings (bool): Whether to normalize the embeddings.

        Returns:
            np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim]).
        """
        return np.array(self.model.encode(texts,
                                          normalize_embeddings=normalize_embeddings,
                                          convert_to_numpy=True))


class EmbeddingModelWithCorrection(EmbeddingModel):
    """
    The EmbeddingModelWithCorrection class is used to convert text inputs into
    vector embeddings, and apply a correction matrix to the embeddings.
    """
    def __init__(self, model_name, correction_matrix):
        """
        Initialize the embedding model with a correction matrix.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            correction_matrix (np.ndarray): A correction matrix to apply to the embeddings.
        """
        super().__init__(model_name)
        self.correction_matrix = correction_matrix

    def encode(self, texts, normalize_embeddings=True):
        """
        Converts a list of text inputs into vector embeddings and applies a correction matrix.

        Args:
            texts (list of str): List of text inputs.
            normalize_embeddings (bool): Whether to normalize the embeddings.

        Returns:
            np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim]).
        """
        embeddings = super().encode(texts, normalize_embeddings)
        if self.correction_matrix is not None:
            embeddings = np.dot(embeddings, self.correction_matrix)
        return embeddings
