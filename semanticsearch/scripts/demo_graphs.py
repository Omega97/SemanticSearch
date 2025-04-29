import os
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.reranking import Reranker
from semanticsearch.src.evaluation import SemanticRetrievalEvaluator


# List of suggested embedding model names (typically Hugging Face identifiers)
EMBEDDING_MODEL_NAMES = [
    # General Purpose & Performance Balance
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",

    # Higher Performance (Larger Models)
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-large",
    "thenlper/gte-base",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Tuned for QA/Search

    # Multilingual
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# List of suggested reranker model names (typically Hugging Face cross-encoder identifiers)
RERANKER_MODEL_NAMES = [
    # High Performance
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-base",
    "mixedbread-ai/mxbai-rerank-large-v1",  # Another recent strong reranker

    # MS MARCO Fine-tuned (Common for Search)
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-electra-base",
]


def test_evaluation(root_dir='..\\..\\data',
                    data_name='yahoo_2000.txt',
                    emb_model_name="all-MiniLM-L6-v2",
                    rerank_model_name: str | None = "BAAI/bge-reranker-large",
                    k_top_docs=10,
                    n_queries=100,
                    chunking_enabled=False,
                    chunk_size=2000,
                    max_n_chunks=10,
                    chunk_overlap=50,
                    show=True,
                    ):
    """
    Test performance of the Semantic Retrieval system.
    """
    print()
    print('data directory:', data_name)
    print('embedding model:', emb_model_name)
    print('reranker model:', rerank_model_name)
    data_name, ext = os.path.splitext(data_name)
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
    plot_name += f'_{data_name}'
    plot_name += f'_{n_queries}_lines'
    plot_name = plot_name.replace('/', '-')
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

    evaluator = SemanticRetrievalEvaluator(embedding_model,
                                           reranker,
                                           tsv_path=tsv_path,
                                           data_dir=data_dir,
                                           k=k_top_docs,
                                           show=show)

    evaluator.run(n_queries, save_path)


def main(root_dir='..\\..\\data', show=False):
    data_names = os.listdir(os.path.join(root_dir, 'test_dataset'))

    # No reranking
    # for i in range(len(data_names)):
    #     test_evaluation(root_dir=root_dir,
    #                     data_name=data_names[i],
    #                     emb_model_name=EMBEDDING_MODEL_NAMES[0],
    #                     rerank_model_name=None,
    #                     n_queries=1000,
    #                     chunking_enabled=False,
    #                     show=show)

    # With reranking
    for i in range(len(data_names)):
        test_evaluation(root_dir=root_dir,
                        data_name=data_names[i],
                        emb_model_name=EMBEDDING_MODEL_NAMES[0],
                        rerank_model_name=RERANKER_MODEL_NAMES[0],
                        n_queries=10,
                        chunking_enabled=False,
                        show=show)


if __name__ == '__main__':
    main()
