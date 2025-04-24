from semanticsearch.src.semantic_retrieval import SemanticRetrieval
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.reranking import Reranker
from semanticsearch.src.evaluation import SemanticRetrievalEvaluator


def test_semantic_retrieval(root_dir='..\\data\\raw', k=10):

    # Load Embedding model
    embedding_model = EmbeddingModel()

    # Load Reranking model
    reranking_model = Reranker(chunking_enabled=True,
                               chunk_size=1000,
                               max_n_chunks=10,
                               chunk_overlap=50)

    # Init Semantic Retrieval app
    app = SemanticRetrieval(root_dir,
                            embedding_model=embedding_model,
                            reranking_model=reranking_model)

    # Example of recommendation
    query = 'Where have Dinosaurs been found?'
    # query = 'An ancient board game played in Asia'
    # query = 'What weapons were used in WWII?'
    app.recommend(query, k=k)


def test_evaluation(tsv_path='..\\data\\test_dataset\\wikiqa_1820.tsv',
                    data_dir='..\\data\\eval_wikiqa_1820',
                    k=10, n_max_rows=10):
    evaluator = SemanticRetrievalEvaluator(tsv_path=tsv_path, data_dir=data_dir, k=k)
    evaluator.run(n_max_rows)


if __name__ == '__main__':
    # test_semantic_retrieval()
    test_evaluation()
