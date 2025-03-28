print('Importing...')
import os.path
import torch
import numpy as np
from semanticsearch.src.training_data import load_tsv
from semanticsearch.src.embedding import EmbeddingModel


def compute_embeddings(data_dir_path, model, batch_size=300):
    """Compute embeddings"""
    print('Computing embeddings...')
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

    rows = len(text_queries)
    for i in range(0, rows, batch_size):
        print(f'{i:4}/{rows}')
        batch = text_queries[i:i + batch_size]
        x = model.encode(batch)

        if i == 0:
            query_embeddings = x
            doc_embeddings = x
        else:
            query_embeddings = np.concatenate((query_embeddings, x))
            doc_embeddings = np.concatenate((doc_embeddings, x))
        if i > 1:
            break

    return query_embeddings, doc_embeddings

def main(data_dir_path='..\\data\\training_dataset'):
    print('Loading model...')
    model = EmbeddingModel()
    t = compute_embeddings(data_dir_path, model, batch_size=300)


if __name__ == '__main__':
    main()
