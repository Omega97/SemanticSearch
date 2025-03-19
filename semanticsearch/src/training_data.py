"""This file contains a class that is used to load the training data
from the data directory."""
import os
import csv
import numpy as np


def load_tsv(filepath, delimiter='\t') -> np.ndarray:
    """
    Loads a TSV file into an N*2 numpy array of query-document pairs.
    Assumes that the column names are not included in the file.

    :param filepath: The path to the TSV file.
    :param delimiter: The delimiter character used in the TSV file.
    :return: A numpy array of query-document pairs.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        data = []
        for row in reader:
            data.append(row)
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
                self.data[key] = load_tsv(os.path.join(self.data_dir, file))

    def get_names(self):
        """returns the names of the data files"""
        return tuple(self.data.keys())

    def get_queries_and_docs(self, name):
        """returns the queries and documents from the specified data file"""
        queries = self.data[name][:, 0]
        docs = self.data[name][:, 1]
        return queries, docs
