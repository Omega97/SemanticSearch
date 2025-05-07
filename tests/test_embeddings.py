from semanticsearch.src.embedding import EmbeddingModel, Embeddings
from semanticsearch.src.database import Database


def test_model():
    """Example usage of EmbeddingModel"""
    model = EmbeddingModel()
    sample_texts = ["Artificial Intelligence is evolving.",
                    "Machine learning enables predictions."]
    embeddings = model.encode(sample_texts)

    print("Embedding shape:", embeddings.shape)
    print("Sample embedding:", embeddings)


def test_embeddings(root_directory="..\\data\\raw",
                    model_name="all-MiniLM-L6-v2"):  # all-MiniLM-L6-v2,
    """Example usage of Embeddings"""
    db = Database(root_directory)
    emb_model = EmbeddingModel(model_name=model_name)
    embeddings = Embeddings(dir_path=root_directory,
                            database=db,
                            embedding_model=emb_model)
    print(embeddings.get_names())


if __name__ == "__main__":
    # test_model()
    test_embeddings()
