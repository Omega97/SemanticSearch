from semanticsearch.scripts.compute_embeddings import compute_embeddings
from semanticsearch.src.knn_search import knn_search
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel


class PageRecommender:
    def __init__(self, data_path='data/raw/', emb_file='data/embeddings.json', k=3):
        """
        Initializes the PageRecommender system.

        :param data_path: Path to the database files.
        :param emb_file: Path to the stored embeddings file.
        :param k: Number of recommendations to return.
        """
        print('Loading database...')
        self.db = Database(data_path)
        print('Loading model...')
        self.emb_model = EmbeddingModel()
        self.emb_file = emb_file
        self.k = k
        print('Preprocessing...')
        self.preprocessing()

    def preprocessing(self):
        """Compute embeddings if needed"""
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


if __name__ == "__main__":
    recommender = PageRecommender(k=3)
    query = 'Diamonds are found mostly in Africa.'  # Example query
    recommender.recommend(query)
