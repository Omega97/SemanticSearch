import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.ranking import count_smaller_than_diagonal, Performance
from semanticsearch.src.training_data import TrainingData
from semanticsearch.src.embedding import EmbeddingModel


def test_1():
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    B = np.array([[1, 3],
                  [1, 2.5],
                  [3, 4]])

    result = count_smaller_than_diagonal(A, B)
    print(result)
    print(np.median(result))


def test_2(k=5):
    # Load the training data
    training_data = TrainingData('..\\data\\training_dataset')
    name = training_data.get_names()[1]
    print(f'Pulling data from {name}...')
    queries, documents = training_data.get_queries_and_docs(name)

    # Initialize the embedding model
    model = EmbeddingModel()
    print(f'model: {model.model_name}')
    print('Computing performance...')

    # Compute the performance of the model
    performance = Performance(queries, documents, max_length=1000)
    n = performance.get_n_queries()
    c = performance.compute_counts(model)
    counts = [np.sum(c == i) for i in range(k)]
    misses = n - np.sum(counts)

    # Plot bar plot of counts
    plt.bar(range(1, k+1), counts, color='blue', alpha=0.7, label='Counts')
    plt.bar(range(k+1, k+2), [misses], color='red', alpha=0.7, label='Misses')

    # add text labels on the bars
    for i in range(k):
        if counts[i] > 0:
            plt.text(i+1, counts[i] + 0.5, f'{counts[i]/n:.1%}', ha='center', va='bottom')
    plt.text(k+1, misses + 0.5, f'{misses/n:.1%}', ha='center', va='bottom')
    plt.legend()
    plt.xlabel('Ranking')
    plt.ylabel('Frequency')
    plt.title(f'Performance of {model.model_name} (on {n} queries)')
    plt.show()


if __name__ == '__main__':
    # test_1()
    test_2()
