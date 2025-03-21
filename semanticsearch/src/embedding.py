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
- stsb-bert-base (large)
- stsb-roberta-base (large)
- stsb-roberta-large (very large)
- stsb-mpnet-base-v2 (very large)
- stsb-mpnet-large-v2 (very large)
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True):
        """
        Converts a list of text inputs into vector embeddings.

        Args:
            texts (list of str): List of text inputs.

        Returns:
            np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim]).
        """
        return np.array(self.model.encode(texts,
                                          normalize_embeddings=normalize_embeddings,
                                          convert_to_numpy=True))
