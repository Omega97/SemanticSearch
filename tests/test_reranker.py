import numpy as np
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semanticsearch.src.reranking import Reranker


def test_1():
    query = 'what is panda'
    docs = [
        'hi',
        'panda is turtle',
        'galapagos',
        'panda panda panda panda panda bla bla bla il cielo è blu come l\'uranio impoverito suscita reazioni esilaranti. COme state voi? Me lo chiedo? Vittorio Emanuele secondo è mio figlio',
        'a panda is a panda',
        'pandas is a python library to manipulate databases',
        'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.',
        'The word panda was borrowed into English from French, but no conclusive explanation of the origin of the French word panda has been found.']

    r = Reranker(chunking_enabled=True)
    docs, scores, chunks = r.doc_rerank(query, docs)

    for i in range(len(docs)):
        print(f'\n---POSITION {i}:---')
        print(f'score = {scores[i]:.3f}')
        print(f'doc = {docs[i][:100]}')
        print(f'chunk = {chunks[i]}')


def test_chunk_splitter(path='..\\data\\raw\\Wikipedia\\Chess.txt',
                        chunk_size=2_000, chunk_overlap=0):
    """
    Test RecursiveCharacterTextSplitter
    :param chunk_size: max character per chunk
    :param chunk_overlap: overlap between chunks to preserve context
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        long_document = f.read()
    # long_document = long_document[:10_000]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_text(long_document)
    for i in range(len(chunks)):
        print(f'\n-------- Chunk {i} ---------')
        print(chunks[i])

    lengths = np.array([len(s) for s in chunks])
    plt.title(f'Chunk length ({len(chunks)} chunks)')
    plt.hist(lengths)
    plt.xlabel('length')
    plt.xlim(0, None)
    plt.show()


if __name__ == '__main__':
    # test_1()
    test_chunk_splitter()
