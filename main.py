from semanticsearch.scripts.compute_embeddings import compute_embeddings
from semanticsearch.src.knn_search import knn_search
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel


def main(k=3):
    # Compute embeddings if needed
    compute_embeddings()

    # Load the database and embeddings
    db = Database('data/raw/')
    emb_model = EmbeddingModel()
    emb_file = 'data/embeddings.json'

    # Write here the query
    # query = 'Dinosaur bones discovered in the Sahara Desert.'
    # query = 'Science was invented in the 17th century.'
    query = 'Diamonds are found mostly in Africa.'

    # Get recommendations
    recommendations = knn_search(query, emb_model, emb_file, k=k)
    print(f'\n\nQuery: {query}\n')
    for i in range(k):
        file_name = recommendations[i]
        doc = db.get_document(file_name)
        print(f'\n\n{doc[:100]}...')


if __name__ == "__main__":
    main()
