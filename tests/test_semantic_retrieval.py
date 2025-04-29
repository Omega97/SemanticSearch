import os
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


def test_evaluation(root_dir='..\\data',
                    data_name='yahoo_2000',
                    emb_model_name="all-MiniLM-L6-v2",
                    rerank_model_name="BAAI/bge-reranker-large",
                    k=10,
                    n_max_rows=100,
                    chunking_enabled=False,
                    chunk_size=2000,
                    max_n_chunks=10,
                    chunk_overlap=50,
                    ):
    """
    Test performance of the Semantic Retrieval system.
    """
    tsv_path = f'{root_dir}\\test_dataset\\{data_name}.tsv'
    data_dir = f'{root_dir}\\eval\\{data_name}'
    img_path = f'{root_dir}\\plots'
    plot_name = f'emb_{emb_model_name}'
    if rerank_model_name is not None:
        plot_name += f'_rerank_{rerank_model_name}'
        if chunking_enabled:
            plot_name += f'chunk-size_{chunk_size}'
            plot_name += f'max-chunks_{max_n_chunks}'
            plot_name += f'overlap_{chunk_overlap}'
    plot_name.replace('/', '-')
    save_path = f'{root_dir}\\plots\\{plot_name}.png'

    os.makedirs(img_path, exist_ok=True)

    # Init Embedding Model
    embedding_model = EmbeddingModel()

    # Init Reranker
    if rerank_model_name is None:
        reranker = None
    else:
        reranker = Reranker(chunking_enabled=chunking_enabled, chunk_size=chunk_size,
                            max_n_chunks=max_n_chunks, chunk_overlap=chunk_overlap, verbose=False)

    evaluator = SemanticRetrievalEvaluator(embedding_model, reranker,
                                           tsv_path=tsv_path, data_dir=data_dir, k=k)

    evaluator.run(n_max_rows, save_path)


if __name__ == '__main__':
    # test_semantic_retrieval()
    test_evaluation()
