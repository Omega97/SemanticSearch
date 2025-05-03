"""
In this demo we retrieve documents from a given folder
based on the similarity to a given query.
"""
from semanticsearch.src.semantic_retrieval import SemanticRetrieval
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.reranking import Reranker


def semantic_retrieval_demo(root_dir='..\\..\\data\\raw', k=10):

    # Load Embedding model
    embedding_model = EmbeddingModel()

    # Load Reranking model
    reranking_model = Reranker(chunking_enabled=False)

    # Init Semantic Retrieval app
    app = SemanticRetrieval(root_dir,
                            embedding_model=embedding_model,
                            reranking_model=reranking_model)

    # Recommendation
    while True:
        query = input('\n>>> ')
        if not query:
            break
        app.recommend(query, k=k)


if __name__ == '__main__':
    semantic_retrieval_demo()
