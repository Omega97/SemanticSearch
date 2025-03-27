from time import time
import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.embedding import EmbeddingModel


def test_model(path="..\\data\\training_dataset\\foreign_english.tsv",
               length=1000, nx=30):
    print('Loading model...')
    model = EmbeddingModel()

    print('Computing...')
    x_ = np.arange(nx) + 1
    y_ = []

    for x in x_:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        sample_texts = [text[length*i:length*(i+1)] for i in range(x)]
        t = time()
        embeddings = model.encode(sample_texts)
        t = (time() - t)/x * 1000
        print(f'{t:.2f}')
        y_.append(t)

    plt.title('Time to process a string')
    plt.plot(x_, y_)
    plt.xlabel('#inputs')
    plt.ylabel('time [ms]')
    plt.show()


if __name__ == "__main__":
    test_model()

