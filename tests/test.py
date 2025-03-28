print('Importing...')
import os.path
import torch
import numpy as np
from semanticsearch.src.training_data import load_tsv
from semanticsearch.src.embedding import EmbeddingModel


def compute_embeddings(data_dir_path, model, batch_size=300, n_lines=None):
    """Compute embeddings"""
    names = [name for name in os.listdir(data_dir_path) if name.endswith('.tsv')]

    # load text
    text_queries = []
    text_docs = []

    for name in names:
        print(f'loading {name}')
        path = os.path.join(data_dir_path, name)
        queries, docs = load_tsv(path)
        text_queries += queries
        text_docs += docs

    query_embeddings = None
    doc_embeddings = None

    print('Computing embeddings...')
    rows = len(text_queries)
    for i in range(0, rows, batch_size):

        batch = text_queries[i:i + batch_size]
        x = model.encode(batch)

        if i == 0:
            query_embeddings = x
            doc_embeddings = x
        else:
            query_embeddings = np.concatenate((query_embeddings, x))
            doc_embeddings = np.concatenate((doc_embeddings, x))

        print(f'{len(query_embeddings):4}/{rows}')
        if len(query_embeddings) >= n_lines:
            break

    return query_embeddings, doc_embeddings


def main(data_dir_path='..\\data\\training_dataset'):
    print('Loading model...')
    model = EmbeddingModel()
    query_embeddings, doc_embeddings = compute_embeddings(data_dir_path, model,
                                                          batch_size=300, n_lines=1000)
    print(doc_embeddings.shape)


if __name__ == '__main__':
    main()
