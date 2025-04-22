import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.evaluation import PerformanceEvaluator
from semanticsearch.src.training_data import TrainingData
from semanticsearch.src.embedding import EmbeddingModel


def plot_results(rank_counts, misses, n_columns, n_queries, title=''):
    # Plot bar plot of counts
    plt.bar(range(1, n_columns + 1), rank_counts, color='blue', alpha=0.7, label='Hits')
    plt.bar(range(n_columns + 1, n_columns + 2), [misses], color='red', alpha=0.7, label='Misses')

    # add text labels on the bars
    for i in range(n_columns):
        if rank_counts[i] > 0:
            plt.text(i + 1, rank_counts[i] + 0.5, f'{rank_counts[i] / n_queries:.1%}', ha='center', va='bottom')
    plt.text(n_columns + 1, misses + 0.5, f'{misses / n_queries:.1%}', ha='center', va='bottom')
    plt.legend()
    plt.xlabel('Recall@n')
    plt.ylabel('Counts')

    y_max = max(max(rank_counts), misses)
    plt.ylim(0, y_max * 1.1)
    plt.xticks(range(1, n_columns + 2), [f'{i}' for i in range(1, n_columns + 1)] + [f'{n_columns + 1}+'])

    plt.title(title)


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
    print(f'model: {model}')
    print('\nComputing performance...')

    # Compute the performance of the model
    performance = PerformanceEvaluator(queries, documents)
    n_queries = performance.get_n_queries()

    # Get the output of the model
    counts = performance.evaluate(model)

    rank_counts = [np.sum(counts == i) for i in range(n_columns)]
    misses = n_queries - np.sum(rank_counts)
    print(f'#queries = {n_queries}')
    print(f'recall = {rank_counts}')
    print(f'misses = {misses}')
    print(f'Number of queries: {n_queries}')

    plot_results(rank_counts, misses, n_columns, n_queries,
                 title=f'Performance of {model.model_name} \n({n_queries} queries form {doc_name})')

    plt.show()


if __name__ == '__main__':
    test_performance()
