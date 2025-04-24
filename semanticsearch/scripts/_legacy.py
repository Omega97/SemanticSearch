#
# class EmbeddingModelWithCorrection(EmbeddingModel):
#     """
#     The EmbeddingModelWithCorrection class is used to convert text inputs into
#     vector embeddings, and apply a correction matrix to the embeddings.
#     """
#     def __init__(self, model_name, correction_matrix):
#         """
#         Initialize the embedding model with a correction matrix.
#
#         Args:
#             model_name (str): The name of the SentenceTransformer model to use.
#             correction_matrix (np.ndarray): A correction matrix to apply to the embeddings.
#         """
#         super().__init__(model_name)
#         self.correction_matrix = correction_matrix
#
#     def encode(self, texts, normalize_embeddings=True):
#         """
#         Converts a list of text inputs into vector embeddings and applies a correction matrix.
#
#         Args:
#             texts (list of str): List of text inputs.
#             normalize_embeddings (bool): Whether to normalize the embeddings.
#
#         Returns:
#             np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim]).
#         """
#         embeddings = super().encode(texts, normalize_embeddings)
#         if self.correction_matrix is not None:
#             embeddings = np.dot(embeddings, self.correction_matrix)
#         return embeddings
