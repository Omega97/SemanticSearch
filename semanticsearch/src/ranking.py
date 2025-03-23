import torch
import numpy as np
from semanticsearch.src.embedding import EmbeddingModel


def count_smaller_than_diagonal(A, B):
    """
    Calculates the distance matrix between vectors in A and B, and counts
    how many elements in each row are strictly smaller than the corresponding
    diagonal element.

    Args:
        A: numpy array (N x M matrix).
        B: numpy array (N x M matrix).

    Returns:
        A list of counts, where each count represents the number of elements
        in the corresponding row of the distance matrix that are strictly
        smaller than the diagonal element.
    """
    # Calculate distance matrix
    dist = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)

    # Get diagonal elements
    diagonal = np.diagonal(dist)

    # Create a boolean mask for elements smaller than the diagonal
    mask = dist < diagonal[:, np.newaxis]

    # Count the number of True values in each row of the mask
    counts = np.sum(mask, axis=1)

    return counts


def recall_at_k(counts, k=1):
    """Returns the frequency of when the count is less than or equal to k."""
    return np.mean(counts <= k)


def compute_recall_at_k(A, B, k=1):
    """Computes the recall@k for the given arrays A and B."""
    counts = count_smaller_than_diagonal(A, B)
    return recall_at_k(counts, k)


def compute_embeddings(model: EmbeddingModel, queries, documents):
    """Computes embeddings for queries and documents using the given model.

    Args:
        model: EmbeddingModel object.  (EmbeddingModel)
        queries: list of queries.   (list of strings)
        documents: list of documents.   (list of strings)
    """
    query_embeddings = torch.tensor(model.encode(queries))
    document_embeddings = torch.tensor(model.encode(documents))
    return query_embeddings, document_embeddings


class PerformanceEvaluator:
    """
    This class is used to compute the performance of a model on a set of queries
    and documents. The performance is measured by counting how many elements in
    each row of the distance matrix are strictly smaller than the corresponding
    diagonal element.
    """
    def __init__(self, queries, documents, score_counts=recall_at_k, max_length=None):
        """
        Initializes the Performance object with the given queries, documents,
        and score_counts function.
        :param queries: np.ndarray list of queries
        :param documents: np.ndarray list of documents
        :param score_counts: function that computes the performance score. Takes as
            input a list of counts and an integer k, and returns the frequency of when
            the count is less than or equal to k.
        :param max_length: maximum number of elements to consider in the queries and documents
        """
        self.queries = queries
        self.documents = documents
        self.score_counts = score_counts
        self.max_length = max_length
        if max_length is not None:
            self.queries = self.queries[:max_length]
            self.documents = self.documents[:max_length]
        self.query_embeddings = None
        self.document_embeddings = None

    def set_embeddings(self, query_embeddings, document_embeddings):
        """Sets the query and document embeddings."""
        self.query_embeddings = query_embeddings
        self.document_embeddings = document_embeddings

    def compute_embeddings(self, model):
        """
        Computes the embeddings for the queries and documents using the given model.
        :param model: model to evaluate
        """
        query_embeddings, document_embeddings = compute_embeddings(model, self.queries, self.documents)
        self.set_embeddings(query_embeddings, document_embeddings)

    def get_embeddings(self):
        """Returns the query and document embeddings."""
        if self.query_embeddings is None or self.document_embeddings is None:
            raise ValueError('Embeddings have not been computed yet.')
        return self.query_embeddings, self.document_embeddings

    def compute_counts(self, model):
        """
        Computes the counts of elements smaller than the diagonal for the given model.
        :param model: model to evaluate
        :return: the counts of elements smaller than the diagonal
        """
        if self.query_embeddings is None or self.document_embeddings is None:
            self.compute_embeddings(model)
        return count_smaller_than_diagonal(self.query_embeddings, self.document_embeddings)

    def compute_score(self, model, k=1):
        """
        Computes the performance of the given model on the queries and documents.
        :param model: model to evaluate
        :param k: threshold for counting the number of elements smaller than k
        :return: the performance score
        """
        counts = self.compute_counts(model)
        return self.score_counts(counts, k)

    def get_n_queries(self):
        """Returns the number of queries."""
        return len(self.queries)
