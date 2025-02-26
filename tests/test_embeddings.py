from semanticsearch.src.embedding import EmbeddingModel


def test_model():
    model = EmbeddingModel()
    sample_texts = ["Artificial Intelligence is evolving.",
                    "Machine learning enables predictions."]
    embeddings = model.encode(sample_texts)

    print("Embedding shape:", embeddings.shape)
    print("Sample embedding:", embeddings)


if __name__ == "__main__":
    test_model()
