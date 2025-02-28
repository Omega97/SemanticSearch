from semanticsearch.scripts.compute_embeddings import compute_embeddings
from semanticsearch.src.knn_search import knn_search
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel


def main():
    # Compute embeddings if needed
    compute_embeddings()

    # Load the database and embeddings
    db = Database('data/raw/')
    emb_model = EmbeddingModel()
    emb_file = 'data/embeddings.json'

    while True:
        # Write here the query
        print('\n')
        query = ''
        while not query:
            query = input('Enter your query: ')

        # Get recommendations
        recommendations = knn_search(query, emb_model, emb_file, k=4)
        file_name = recommendations[0]
        doc = db.get_document(file_name)
        print(f'\n\n{doc[:100]}...')
        print(f'See also: {recommendations[1:]}')


if __name__ == "__main__":
    main()
