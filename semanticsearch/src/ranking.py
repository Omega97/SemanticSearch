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


def mean_score(counts, k=3):
    """Returns the frequency of when the count is less than or equal to k."""
    return np.mean(counts <= k)


def compute_embeddings(model: EmbeddingModel, queries, documents):
    """Computes embeddings for queries and documents using the given model.

    Args:
        model: EmbeddingModel object.  (EmbeddingModel)
        queries: list of queries.   (list of strings)
        documents: list of documents.   (list of strings)
    """
    query_embeddings = model.encode(queries)
    document_embeddings = model.encode(documents)
    return query_embeddings, document_embeddings


class Performance:
    """
    This class is used to compute the performance of a model on a set of queries
    and documents. The performance is measured by counting how many elements in
    each row of the distance matrix are strictly smaller than the corresponding
    diagonal element.
    """
    def __init__(self, queries, documents, score_counts=mean_score, max_length=None):
        """
        Initializes the Performance object with the given queries, documents,
        and score_counts function.
        :param queries: list of queries
        :param documents: list of documents
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

    def compute_counts(self, model):
        """
        Computes the counts of elements smaller than the diagonal for the given model.
        :param model: model to evaluate
        :return: the counts of elements smaller than the diagonal
        """
        query_embeddings, document_embeddings = compute_embeddings(model, self.queries, self.documents)
        return count_smaller_than_diagonal(query_embeddings, document_embeddings)

    def compute_score(self, model, k=3):
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
