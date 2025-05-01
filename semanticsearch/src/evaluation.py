import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from semanticsearch.src.semantic_retrieval import SemanticRetrieval
from semanticsearch.src.embedding import EmbeddingModel


def compute_counts(A, B):
    """
    Calculates the distance matrix between vectors in A and B, and counts
    how many elements in each row are strictly smaller than the corresponding
    diagonal element.

    Args:
        A: numpy array (N x M).
        B: numpy array (N x M).

    Returns:
        A float: A list of counts, where each count represents the number of elements
            in the corresponding row of the distance matrix that are strictly
            smaller than the diagonal element.
    """
    N = A.shape[0]
    counts = np.zeros(N, dtype=int)

    for i in range(N):
        # Compute distance of A[i] to each row in B
        distances = np.linalg.norm(A[i] - B, axis=1)
        # Diagonal element: distance between A[i] and B[i]
        diag = distances[i]
        # Count how many distances are strictly less than diag
        counts[i] = np.sum(distances < diag)

    return counts


def recall_at_k(counts, k=1):
    """Returns the frequency of when the count is less than or equal to k."""
    return np.mean(counts <= k)


def compute_recall_at_k(A, B, k=1):
    """Computes the recall@k for the given arrays A and B."""
    counts = compute_counts(A, B)
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
    def __init__(self, queries, documents):
        """
        Initializes the Performance object with the given queries, documents,
        and score_counts function.
        :param queries: np.ndarray list of queries
        :param documents: np.ndarray list of documents
        """
        self.queries = queries
        self.documents = documents
        self.query_embeddings = None
        self.document_embeddings = None
        self._counts = None

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

    def evaluate(self, model):
        """
        Computes the counts of elements smaller than the diagonal for the given model.
        Forces re-compute of the self._counts variable.
        :param model: model to evaluate
        :return: the counts of elements smaller than the diagonal
        """
        if self.query_embeddings is None or self.document_embeddings is None:
            self.compute_embeddings(model)
        self._counts = compute_counts(self.query_embeddings, self.document_embeddings)
        return self._counts

    def compute_recall_at_k(self, model, k=1):
        """
        Computes the performance of the given model on the queries and documents.
        :param model: model to evaluate
        :param k: threshold for counting the number of elements smaller than k
        :return: the performance score
        """
        if self._counts is None:
            self.evaluate(model)
        return recall_at_k(self._counts, k)

    def get_n_queries(self):
        """Returns the number of queries."""
        return len(self.queries)


class SemanticRetrievalEvaluator:
    def __init__(self, embedding_model, reranker, tsv_path: str,
                 data_dir: str, k: int=10, show=True):
        """
        Initialize the evaluator with dataset and configuration.

        Args:
            embedding_model: embedding model
            reranker = reranker
            tsv_path (str): Path to the .tsv file with columns [query, document].
            data_dir (str): Directory to store document files.
            k (int): Top-k documents to consider during evaluation.
        """
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.tsv_path = tsv_path
        self.data_dir = data_dir
        self.k = k
        self.show = show
        self.df = None
        self.rank_counts = [0] * k
        self.misses = 0
        self.n_queries = 0

    def _prepare_documents(self):
        # Load the TSV file with no header and assign column names manually
        self.df = pd.read_csv(self.tsv_path, sep='\t', header=None, names=['query', 'document'])
        os.makedirs(self.data_dir, exist_ok=True)

        for i, row in self.df.iterrows():
            doc_path = os.path.join(self.data_dir, f"{i}.txt")
            if not os.path.exists(doc_path):
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(row['document'])

    def _init_retrieval_system(self):
        return SemanticRetrieval(
            root_dir=self.data_dir,
            embedding_model=self.embedding_model,
            reranking_model=self.reranker,
            verbose=False
        )

    def _evaluate_queries(self, retrieval_system, n_queries=None):

        if n_queries is None:
            iterable = self.df.iterrows()
            length = len(self.df)
        else:
            iterable = self.df.head(n_queries).iterrows()
            length = min(n_queries, len(self.df))

        for i, row in tqdm(iterable, total=length, leave=True):
            self.n_queries += 1
            query = row['query']
            correct_doc = f"{i}.txt"

            ranked_paths = retrieval_system.recommend(query, k=self.k)

            try:
                rank = ranked_paths.index(correct_doc)
                self.rank_counts[rank] += 1
            except ValueError:
                self.misses += 1

    def _plot_results(self, title='Semantic Retrieval Evaluation', save_path=None):
        """
        Plots the evaluation results and optionally saves the plot to a file.

        Parameters:
            title (str): The title of the plot.
            save_path (str, optional): The file path where png the plot should be saved.
                If None, the plot is not saved.
        """
        plt.bar(range(1, self.k + 1), self.rank_counts, color='blue', alpha=0.7, label='Hits')
        plt.bar([self.k + 1], [self.misses], color='red', alpha=0.7, label='Misses')

        for i in range(self.k):
            if self.rank_counts[i] > 0:
                plt.text(i + 1, self.rank_counts[i] + 0.5, f'{self.rank_counts[i] / self.n_queries:.1%}',
                         ha='center', va='bottom')
        plt.text(self.k + 1, self.misses + 0.5, f'{self.misses / self.n_queries:.1%}', ha='center', va='bottom')

        plt.legend()
        plt.xlabel('Recall@n')
        plt.ylabel('Counts')
        plt.ylim(0, max(max(self.rank_counts), self.misses) * 1.1)
        plt.xticks(range(1, self.k + 2), [str(i) for i in range(1, self.k + 1)] + [f'{self.k + 1}+'])
        plt.title(title)

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution and tight layout

        if self.show:
            plt.show()
        else:
            plt.close()

    def run(self, n_queries=None, save_path=None, title=None):
        print("Preparing documents...")
        self._prepare_documents()

        print("Initializing retrieval system...")
        retrieval_system = self._init_retrieval_system()

        print("Evaluating queries...")
        self._evaluate_queries(retrieval_system, n_queries)

        print("\nEvaluation complete.")
        print(f"Total queries: {self.n_queries}")
        print(f"Misses (correct doc not in top-{self.k}): {self.misses}")

        self._plot_results(title=title,
                           save_path=save_path)
