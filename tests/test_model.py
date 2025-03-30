import numpy as np
import matplotlib.pyplot as plt
print('Importing...')
from semanticsearch.src.embedding import EmbeddingModel


def test_model(path='..\\data\\raw\\Wikipedia\\Chess.txt',
               n_words=1000, window_length=20, clim = 0.15):
    print('Loading model...')
    model = EmbeddingModel()

    print('Computing...')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\n', ' ')

    words = text.split(' ')
    words = [w for w in words[:n_words] if w]

    batch = []
    for i in range(0, len(words), window_length):
        w = words[i: i+window_length]
        text = ' '.join(w)
        batch.append(text)

    mat = model.encode(batch)

    query = "Organized chess arose in the 19th century. Chess competition today is governed internationally by FIDE"
    query_emb = model.encode([query])[0]

    similarity = mat @ query_emb
    similarity -= np.average(similarity)
    similarity /= np.std(similarity)

    for j, i in enumerate(range(0, len(words), window_length)):
        w = words[i: i+window_length]
        print(f'{similarity[j]:+.2f} ', " ".join(w))


if __name__ == "__main__":
    test_model()
