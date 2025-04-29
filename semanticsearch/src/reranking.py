""" semanticsearch/src/reranking.py

Reranking model
"""
from FlagEmbedding import FlagReranker
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semanticsearch.src.misc import cprint


# ----- Default Parameters -----
DEFAULT_RANKER_MODEL = "BAAI/bge-reranker-large"


class Reranker:
    """
    Sorts the documents from most to least relevant wrt the query.
    """
    def __init__(self, chunking_enabled=False, chunk_size=2000,
                 max_n_chunks=10, chunk_overlap=50,
                 ranker_model_name=DEFAULT_RANKER_MODEL, verbose=True):
        """
        Initialize Reranker object

        :param chunking_enabled:
        :param chunk_size: max characters per chunk
        :param chunk_overlap: overlap characters between chunks to preserve context
        """
        cprint(f'Loading re-ranking model {ranker_model_name}', 'b')
        self.reranker_model = FlagReranker(ranker_model_name, use_fp16=True)
        self.chunking_enabled = chunking_enabled
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.max_n_chunks = max_n_chunks
        self.verbose = verbose

        self.top_scores = None
        self.top_docs = None
        self.top_chunks = None

    def cprint(self, text, color_code=None):
        if self.verbose:
            cprint(text, color_code)

    def _compute_scores(self, query: str, documents: list):
        """Use reranker_model to compute the scores."""
        pairs = [(query, doc) for doc in documents]
        self.top_scores = np.array(self.reranker_model.compute_score(pairs))

    def _compute_best_chunks(self, query: str, documents: list):
        """
        Compute a list of scores for each chunk of each document. The score
        of the document is score of the best chunk.
        """
        # Ensure chunker is available (should be if chunking_enabled was True)
        if not self.chunker:
            raise RuntimeError("Chunker is not initialized. "
                               "Ensure chunking_enabled=True during Reranker initialization.")

        self.top_scores = []
        self.top_chunks = []

        for doc in documents:

            doc_chunks = self.chunker.split_text(doc)

            # Create pairs of (query, chunk) for scoring
            doc_pairs = [(query, chunk) for chunk in doc_chunks]
            if len(doc_pairs) > self.max_n_chunks:
                doc_pairs = doc_pairs[:self.max_n_chunks]

            # Compute scores for all chunks
            print(f'Computing score on {len(doc_pairs)} chunks...')
            doc_scores = np.array(self.reranker_model.compute_score(doc_pairs))

            # Find the index of the highest score
            best_score_index = np.argmax(doc_scores)
            best_chunk = doc_chunks[best_score_index]
            best_score = doc_scores[best_score_index].item()  # Use .item() to get Python float

            self.top_scores.append(best_score)
            self.top_chunks.append(best_chunk)
            txt = best_chunk[:200]
            txt = txt.replace('\n', ' ')
            print(f' {best_score:+.2f}  {txt}...')

        self.top_scores = np.array(self.top_scores)

    def get_sorted_docs_scores(self) -> dict:
        new_ordering = np.argsort(self.top_scores)[::-1]  # Sort from biggest to smallest
        top_docs = [self.top_docs[i] for i in new_ordering]
        top_scores = self.top_scores[new_ordering]
        if self.top_chunks is not None:
            top_chunks = [self.top_chunks[i] for i in new_ordering]
        else:
            top_chunks = None

        return {'docs': top_docs,
                'permutation': new_ordering,
                'scores': top_scores,
                'chunks': top_chunks}

    def doc_rerank(self, query: str, top_documents: list[str]) -> dict:
        """
        Function taking in input the query string and a list of document
        selected with knn. Returns a permutation of the document
        based on a reranking model.
        """
        self.cprint(f'Re-ranking {len(top_documents)} documents...', "b")
        self.top_docs = top_documents

        if not self.chunking_enabled:
            self._compute_scores(query, top_documents)
        else:
            self._compute_best_chunks(query, top_documents)

        return self.get_sorted_docs_scores()

    def __call__(self, query: str, top_documents: list[str]) -> list[int]:
        """
        Re-rank the documents from most relevant to least relevant.
        Returns the permutation that has to be applied on the list of paths/docs,
        where the best doc is at the beginning of the list.
        """
        return self.doc_rerank(query, top_documents)['permutation']