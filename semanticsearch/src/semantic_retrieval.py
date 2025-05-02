""" semanticsearch/src/semantic_retrieval.py

Semantic Retrieval
"""
import numpy as np
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import Embeddings
from semanticsearch.src.misc import cosine_similarity
from semanticsearch.src.misc import cprint


class SemanticRetrieval:
    """
    Given a query, this class returns an ordering of the provided documents from
    most to leas relevant.
    """
    def __init__(self, root_dir, embedding_model, reranking_model, verbose=True, recompute_embeddings=False):
        """
        Initialize the semantic retrieval system
        :param root_dir:
        :param embedding_model:
        :param reranking_model:
        """
        self.root_dir = root_dir
        self.embedding_model = embedding_model
        self.reranking_model = reranking_model
        self.verbose = verbose
        self.recompute_embeddings = recompute_embeddings

        self.database = None
        self.embedding_path = None
        self._preprocessing()

    def cprint(self, text, color_code='w'):
        if self.verbose:
            cprint(text, color_code)

    def _preprocessing(self):
        # Load Database
        self.database = Database(self.root_dir)

        # Load Embeddings
        self.embeddings = Embeddings(dir_path=self.root_dir,
                                     database=self.database,
                                     embedding_model=self.embedding_model,
                                     recompute_embeddings=self.recompute_embeddings)

    def recommend(self, query, k=10):
        """
        Recommend the top-k most relevant documents to a query using embedding similarity,
        followed by reranking. Prints top-k based on similarity before reranking.
        """
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]
        names, embedding_matrix = self.embeddings.to_matrix()

        # Compute cosine similarities
        sims = cosine_similarity(query_embedding, embedding_matrix)
        top_indices = np.argsort(sims)[::-1][:k]
        top_paths = [names[i] for i in top_indices]
        top_docs = [self.database.get_document(name) for name in top_paths]
        top_scores = sims[top_indices]

        # Print similarity-based ranking
        self.cprint("Top results before reranking (by similarity):")
        for i, (path, sim_score) in enumerate(zip(top_paths, top_scores)):
            self.cprint(f"{i + 1:2d}. ({sim_score:+.3f}) — {path}")

        # Rerank based on semantic relevance
        if self.reranking_model is not None:
            reranked = self.reranking_model.doc_rerank(query, top_docs)
            top_scores = reranked['scores']
            top_paths = [top_paths[i] for i in reranked['permutation']]

            # Print reranked results
            self.cprint("Top results after reranking:")
            for i, (path, score) in enumerate(zip(top_paths, top_scores)):
                self.cprint(f"{i + 1:2d}. ({score:+.2f}) — {path}")

        return top_paths
