from FlagEmbedding import FlagReranker
import numpy as np



class Reranker:

    def __init__(self):
        self.reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

    def rerank(self, query: str, top_documents: list[str], k: int | None = None) -> list[str]:
        """Function taking in input the query string and a list of document selected with knn. Returns a permutation of the document based on a reranking model"""
        k = k if k is not None else len(top_documents)

        pairs = [[query, doc] for doc in top_documents]
        scores = np.array(self.reranker_model.compute_score(pairs)) # compute scores
        new_ordering = np.argsort(scores)[::-1]
        new_top_docs = [top_documents[i] for i in new_ordering]
        
        return new_top_docs[:k], [scores[i] for i in new_ordering][:k]