import os
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.reranking import Reranker
from semanticsearch.src.evaluation import SemanticRetrievalEvaluator
import shutil


# List of suggested embedding model names (typically Hugging Face identifiers)
EMBEDDING_MODEL_NAMES = [
    # General Purpose & Performance Balance
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",  # good

    # Higher Performance (Larger Models)
    "BAAI/bge-large-en-v1.5",  # very good
    "BAAI/bge-base-en-v1.5",   # good
    "thenlper/gte-large",  # very good
    "thenlper/gte-base",   # very good
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Tuned for QA/Search, good

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
                    recompute_embeddings=False,
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

    # Plot name
    plot_name = f'{data_name}'
    plot_name += f'_{n_queries}_queries'
    plot_name += f'_emb_{emb_model_name}'
    if rerank_model_name is not None:
        plot_name += f'_rerank_{rerank_model_name}'
        if chunking_enabled:
            plot_name += f'chunk-size_{chunk_size}'
            plot_name += f'max-chunks_{max_n_chunks}'
            plot_name += f'overlap_{chunk_overlap}'
    plot_name = plot_name.replace('/', '-')
    save_path = f'{root_dir}\\plots\\{plot_name}.png'

    # Title
    title = f"Semantic Retrieval Evaluation on {data_name} (Top-{k_top_docs})"
    title += f'\nembedding={emb_model_name}'
    title += f', reranker={rerank_model_name}'

    os.makedirs(img_path, exist_ok=True)

    # Init Embedding Model
    embedding_model = EmbeddingModel(model_name=emb_model_name)

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
                                           show=show,
                                           recompute_embeddings=recompute_embeddings)

    evaluator.run(n_queries, save_path, title=title)


def main(root_dir='..\\..\\data', data_dir='..\\..\\data\\test_dataset', show=False):
    datasets = os.listdir(data_dir)

    # No reranking
    # for i in range(len(data_names)):
    #     test_evaluation(root_dir=root_dir,
    #                     data_name=data_names[i],
    #                     emb_model_name=EMBEDDING_MODEL_NAMES[1],
    #                     rerank_model_name=None,
    #                     n_queries=1000,
    #                     chunking_enabled=False,
    #                     show=show)

    # Change embeddings
    # data_name = 'foreign_foreign_2000'  # 'papers_dataset_924'
    # for i in range(len(EMBEDDING_MODEL_NAMES)):
    #     test_evaluation(root_dir=root_dir,
    #                     data_name=f'{data_name}.tsv',
    #                     emb_model_name=EMBEDDING_MODEL_NAMES[i],
    #                     rerank_model_name=None,
    #                     n_queries=1000,
    #                     chunking_enabled=False,
    #                     show=show,
    #                     recompute_embeddings=True)

    # All datasets
    # for i in range(len(datasets)):
    #     test_evaluation(root_dir=root_dir,
    #                     data_name=datasets[i],
    #                     emb_model_name='thenlper/gte-base',
    #                     rerank_model_name=None, # RERANKER_MODEL_NAMES[0]
    #                     n_queries=200,
    #                     chunking_enabled=False,
    #                     show=show,
    #                     recompute_embeddings=True)

    # With reranking
    test_evaluation(root_dir=root_dir,
                    data_name='golden_answer_2000',
                    emb_model_name='thenlper/gte-base',
                    rerank_model_name=RERANKER_MODEL_NAMES[2],  # None
                    n_queries=200,
                    chunking_enabled=False,
                    show=show,
                    recompute_embeddings=False)


if __name__ == '__main__':
    main()
