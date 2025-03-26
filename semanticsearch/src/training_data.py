"""This file contains a class that is used to load the training data
from the data directory."""
import os
import csv
import numpy as np


# def load_tsv_(filepath, delimiter='\t') -> np.ndarray:
#     """
#     Loads a TSV file into an N*2 numpy array of query-document pairs.
#     Assumes that the column names are not included in the file.
#
#     :param filepath: The path to the TSV file.
#     :param delimiter: The delimiter character used in the TSV file.
#     :return: A numpy array of query-document pairs.
#     """
#     with open(filepath, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f, delimiter=delimiter)
#         data = []
#         for row in reader:
#             data.append(row)
#     return np.array(data)


def load_tsv(filepath, delimiter='\t') -> np.ndarray:
    """
    Loads a TSV file into an N*2 numpy array of query-document pairs.
    Assumes that the column names are not included in the file.
    Warns and skips lines that cause csv.Error during reading.

    :param filepath: The path to the TSV file.
    :param delimiter: The delimiter character used in the TSV file.
    :return: A numpy array of query-document pairs.
    """
    data = []
    line_index = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        while True:
            line_index += 1
            try:
                row = next(reader)
                data.append(row)
            except StopIteration:
                break  # End of file reached
            except csv.Error as e:
                print(f"Warning: CSV error in file {filepath} on line {line_index}: {e}. Skipping.")
                # Read until the next newline to skip the rest of the problematic line
                f.readline()
                continue
    return np.array(data)


class TrainingData:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = dict()
        self._load_data()

    def _load_data(self):
        """Load all the 'tsv' files from the data directory."""
        for file in os.listdir(self.data_dir):
            if file.endswith('.tsv'):
                key = file.split('.')[0]
                try:
                    print(f'Loading {key}')
                    self.data[key] = load_tsv(os.path.join(self.data_dir, file))
                except csv.Error as e:
                    print(f"\nError {e} with file {file}")

    def get_names(self):
        """returns the names of the data files"""
        return tuple(self.data.keys())

    def get_queries_and_docs(self, *names, shuffle=True, max_size=None):
        """
        Returns the queries and documents from the specified data files
        in a shuffled order, and in the specified size.
        """
        queries = []
        docs = []

        # Concatenate the queries and documents from the specified data files
        for name in names:
            if name in self.data:
                data = self.data[name]
                queries.extend(data[:, 0])
                docs.extend(data[:, 1])
            else:
                print(f"Error: {name} not found in data directory.")

        # Shuffle the data
        if shuffle:
            indices = np.random.permutation(len(queries))
            queries = [queries[i] for i in indices]
            docs = [docs[i] for i in indices]

        # Limit the size of the data
        if max_size is not None:
            queries = queries[:max_size]
            docs = docs[:max_size]

        return queries, docs
