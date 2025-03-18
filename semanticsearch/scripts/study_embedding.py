"""
Load a document, and plot how the embedding of the first part of the document
changes as we move through the document.
This plot shows how the embedding changes as you move through the document.
"""
import matplotlib.pyplot as plt
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.database import Database
import os
from time import time
import numpy as np


def compute_embedding_sliding_window(tokens: list, model: EmbeddingModel, window_size=100):
    """ Compute the embedding of a sliding window
    of the document that moves through the document."""
    texts = [" ".join(tokens[i:i + window_size]) for i in range(0, len(tokens), window_size)]
    return model.encode(texts)


def compare_embeddings():

    with open('query_examples.txt', 'r') as file:
        texts = file.read().split('\n')

    texts = [t for t in texts if t]
    texts = [t for t in texts if not t.startswith('#')]

    model = EmbeddingModel()
    mat = model.encode(texts)

    plt.title('Correlation of query embedding')
    plt.imshow(np.cov(mat) * mat.shape[1], cmap='bwr')
    clim = 1.1
    plt.clim(-clim, clim)
    plt.xlabel('Embedding')
    plt.ylabel('Embedding')
    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()


def plot_embedding(path='..\\..\\data\\raw',
                   embedding_length=101, clim=0.15):
    db = Database(path)
    # text = db.get_document(os.path.join(path, 'Wikipedia\\Binary Golay code.txt'))
    text = db.get_document(os.path.join(path, 'Wikipedia\\Tyrannosaurus.txt'))
    print(text[:100])

    tokens = text.split()

    print(f'{len(tokens)} tokens')
    t = time()
    model = EmbeddingModel()
    t = time() - t
    print(f'Computed in {t:.2f} seconds')

    mat = compute_embedding_sliding_window(tokens, model)

    plt.title('Progressive embedding')
    plt.imshow(mat[:, :embedding_length], cmap='bwr')
    plt.clim(-clim, clim)
    plt.xlabel('Embedding')
    plt.ylabel('Number of words')

    # set aspect ratio
    plt.gca().set_aspect('auto')

    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    compare_embeddings()
    # plot_embedding()
