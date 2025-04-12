"""
This file contains a class that evaluates the system's performance
by verifying the ranking of the correct document in the list of
retrieved documents. The higher the rank of the correct document, the
worst the system's performance. The class also contains a method that
calculates the mean reciprocal rank (MRR) of the system's performance.
Todo check if the files are in the correct location
"""
from semanticsearch.scripts.compute_embeddings import compute_embeddings
from semanticsearch.src.knn_search import knn_search
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel


class PageRecommender:
    """
    The PageRecommender system takes a search query and returns the paths
    of the top-k recommended documents from the database based on the
    similarity of the query to the documents' embeddings.
    """
    def __init__(self, data_path, emb_file, k=3):
        """
        :param data_path: Path to the database files (a directory).
        :param emb_file: Path to the stored embeddings file (a .json file).
        :param k: Number of recommendations to return.
        """
        self.data_path = data_path
        self.emb_file = emb_file
        self.k = k
        self.db = None
        self.emb_model = None
        self._load_database()
        self._load_emb_model()
        self._preprocessing()

    def _load_database(self):
        print('Loading database...')
        self.db = Database(self.data_path)

    def _load_emb_model(self):
        print('Loading embedding model...')
        self.emb_model = EmbeddingModel()

    def _preprocessing(self):
        """Compute embeddings if needed"""
        print('Preprocessing...')
        compute_embeddings(self.db, self.emb_model, self.emb_file)

    def get_document(self, file_name):
        """
        Retrieve the file content from the database

        :param file_name: name of the file
        """
        return self.db.get_document(file_name)

    def recommend(self, query):
        """
        Retrieves and prints the top-k recommended documents for a given query.

        :param query: The search query.
        """
        return knn_search(query, self.emb_model, self.emb_file, self.k)
