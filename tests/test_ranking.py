import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.ranking import PerformanceEvaluator
from semanticsearch.src.training_data import TrainingData
from semanticsearch.src.embedding import EmbeddingModel


def test_performance(doc_id=0, k=1):
    # Load the training data
    training_data = TrainingData('..\\data\\training_dataset')
    names = training_data.get_names()
    name = names[doc_id]

    print('\nDocuments:')
    print('\n'.join(names))
    print(f'\nPulling data from {name}...')
    queries, documents = training_data.get_queries_and_docs(name)

    # Initialize the embedding model
    model = EmbeddingModel()
    print(f'model: {model.model_name}')
    print('\nComputing performance...')

    # Compute the performance of the model
    performance = PerformanceEvaluator(queries, documents, max_length=1000)
    n = performance.get_n_queries()
    c = performance.compute_counts(model)
    counts = [np.sum(c == i) for i in range(k)]
    misses = n - np.sum(counts)

    print(misses)
    print(n - performance.compute_recall_at_k(6))

    print(f'Number of queries: {n}')

    # Plot bar plot of counts
    plt.bar(range(1, k+1), counts, color='blue', alpha=0.7, label='Counts')
    plt.bar(range(k+1, k+2), [misses], color='red', alpha=0.7, label='Misses')

    # add text labels on the bars
    for i in range(k):
        if counts[i] > 0:
            plt.text(i+1, counts[i] + 0.5, f'{counts[i]/n:.1%}', ha='center', va='bottom')
    plt.text(k+1, misses + 0.5, f'{misses/n:.1%}', ha='center', va='bottom')
    plt.legend()
    plt.xlabel('Recall@n')
    plt.ylabel('Frequency')
    y_max = max(counts)
    y_max = max(y_max, misses)
    plt.ylim(0, y_max * 1.1)
    plt.xticks(range(1, k+2), [f'{i}' for i in range(1, k+1)] + [f'{k+1}+'])

    plt.title(f'Performance of {model.model_name} (on {n} queries)')
    plt.show()


if __name__ == '__main__':
    # test_count()
    test_performance()
