from semanticsearch.src.knn_search import knn_search
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel


def test_knn_search(k=3):
    db = Database('../data/raw/')
    emb_model = EmbeddingModel()
    emb_file = '../data/embeddings.json'

    # query = 'Dinosaur bones discovered in the Sahara Desert.'
    # query = 'Science was invented in the 17th century.'
    query = 'Diamonds are found mostly in Africa.'
    print(f'\nQuery: {query}\n')

    recommendations = knn_search(query, emb_model, emb_file, k=k)

    for i in range(k):
        file_name = recommendations[i]
        doc = db.get_document(file_name)
        print(f'\n\n{doc[:100]}...')


if __name__ == '__main__':
    test_knn_search()
