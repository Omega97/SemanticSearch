import os
from semanticsearch.src.semantic_retrieval import SemanticRetrieval
from semanticsearch.src.embedding import EmbeddingModel


def create_folder(root_dir='..\\..\\data', name='words', n_words=None):
    data_dir = os.path.join(root_dir, name)
    data_file = os.path.join(root_dir, f'{name}.txt')

    os.makedirs(data_dir, exist_ok=True)

    print('Creating the files...')
    with open(data_file, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')
        if n_words is not None:
            words = words[:n_words]

    for word in words:
        path = os.path.join(data_dir, f'{word}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(word)

    print('Done!')


def words_demo(root_dir='..\\..\\data\\words', k=10):

    # Load Embedding model
    embedding_model = EmbeddingModel()

    # Init Semantic Retrieval app
    app = SemanticRetrieval(root_dir,
                            embedding_model=embedding_model,
                            reranking_model=None)

    # Recommendation
    while True:
        query = input('\n>>> ')
        if not query:
            break
        app.recommend(query, k=k)


if __name__ == '__main__':
    # create_folder(n_words=3000)
    words_demo()
