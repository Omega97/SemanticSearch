import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.ranking import PerformanceEvaluator
from semanticsearch.src.training_data import TrainingData
from semanticsearch.src.embedding import EmbeddingModel


def test_performance(doc_name='msmarco_5001', n_columns=9, max_length=1000):
    """
    Test for system performance
    Doc names:
    * msmarco_5001
    * wikiqa_1820
    * question_golden_answer_5560
    * yahoo_train_2000
    """
    # Load the training data
    training_data = TrainingData('..\\data\\training_dataset')
    names = training_data.get_names()

    if type(doc_name) is int:
        doc_name = names[doc_name]

    print('\nDocuments:')
    for i in range(len(names)):
        print(f'{i}) {names[i]}')
    print(f'\nPulling data from: {doc_name}...')
    queries, documents = training_data.get_queries_and_docs(doc_name, max_size=max_length)
    print(f'{len(queries)} queries')

    # Initialize the embedding model
    model = EmbeddingModel()
    print(f'model: {model.model_name}')
    print('\nComputing performance...')

    # Compute the performance of the model
    performance = PerformanceEvaluator(queries, documents)
    n = performance.get_n_queries()
    counts = performance.compute_counts(model)
    recall_at_k = [np.sum(counts == i) for i in range(n_columns)]
    misses = n - np.sum(recall_at_k)

    print(f'#queries = {n}')
    print(f'recall = {recall_at_k}')
    print(f'misses = {misses}')
    print(f'Number of queries: {n}')

    # Plot bar plot of counts
    plt.bar(range(1, n_columns + 1), recall_at_k, color='blue', alpha=0.7, label='Counts')
    plt.bar(range(n_columns + 1, n_columns + 2), [misses], color='red', alpha=0.7, label='Misses')

    # add text labels on the bars
    for i in range(n_columns):
        if recall_at_k[i] > 0:
            plt.text(i+1, recall_at_k[i] + 0.5, f'{recall_at_k[i]/n:.1%}', ha='center', va='bottom')
    plt.text(n_columns + 1, misses + 0.5, f'{misses / n:.1%}', ha='center', va='bottom')
    plt.legend()
    plt.xlabel('Recall@n')
    plt.ylabel('Frequency')
    y_max = max(recall_at_k)
    y_max = max(y_max, misses)
    plt.ylim(0, y_max * 1.1)
    plt.xticks(range(1, n_columns + 2), [f'{i}' for i in range(1, n_columns + 1)] + [f'{n_columns + 1}+'])

    plt.title(f'Performance of {model.model_name}'
              f'\n({n} queries form {doc_name})')
    plt.show()


if __name__ == '__main__':
    test_performance()
