from FlagEmbedding import FlagReranker
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Reranker:

    def __init__(self, chunking_enabled: bool = False):
        self.reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        self.chunking_enabled = chunking_enabled
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=30,      # max tokens per chunk
            chunk_overlap=5,     # overlap between chunks to preserve context
        )

    def doc_rerank(self, query: str, top_documents: list[str], k: int | None = None) -> list[str]:
        """Function taking in input the query string and a list of document selected with knn. Returns a permutation of the document based on a reranking model"""
        k = k if k is not None else len(top_documents)

        if not self.chunking_enabled:
            pairs = [[query, doc] for doc in top_documents]
            scores = np.array(self.reranker_model.compute_score(pairs))

        elif self.chunking_enabled:
            scores = np.empty(shape=(len(top_documents)))
            best_chunks = [None] * len(top_documents)
            # divide each document into chunks
            # calculate the highest score for chunk
            # the score of the document is the score of the highest scoring chunk
            # possibly return best chunk
            for i in range(len(top_documents)):
                # divide doc in chunks and calculate score for each chunk
                doc = top_documents[i]
                doc_chunks = self.chunker.split_text(doc)
                doc_pairs = [[query, chunk] for chunk in doc_chunks]
                doc_scores = np.array(self.reranker_model.compute_score(doc_pairs))
                
                # document score is best chunk score
                score = np.max(doc_scores).item() # final score of the doc
                scores[i] = score
                # save best chunk
                best_chunks[i] = doc_chunks[np.argmax(doc_scores).item()]

        new_ordering = np.argsort(scores)[::-1]
        new_top_docs = [top_documents[i] for i in new_ordering]

        return new_top_docs[:k], [scores[i] for i in new_ordering][:k], [best_chunks[i] for i in new_ordering[:k]] if 'best_chunks' in locals() else None
    


# text_splitter = RecursiveCharacterTextSplitter(
# chunk_size=500,      # max tokens per chunk
# chunk_overlap=50,     # overlap between chunks to preserve context
# )

# chunks = text_splitter.split_text(long_document)
# for i in range(10):
#     print(f'--------{i}---------')
# print(chunks[i])