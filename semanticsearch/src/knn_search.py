import numpy as np
import json


def knn_search(query, embedding_model, embeddings_file, k=5):
    """
    Finds the top-k most similar documents to the given query using cosine similarity.

    Parameters:
    - query: str, the user query
    - embedding_model: EmbeddingModel instance (used to compute query embeddings)
    - embeddings_file: str, path to the JSON file containing embeddings
    - k: int, number of top results to return

    Returns:
    - List of k most relevant document paths
    """

    # Load stored embeddings
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    # Compute the embedding for the query
    query_embedding = embedding_model.encode([query])[0]

    # Compute cosine similarity between query embedding and document embeddings
    similarities = {}
    for file_path, embedding in embeddings_data.items():
        embedding = np.array(embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities[file_path] = similarity

    # Sort documents by similarity score in descending order
    sorted_results = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    # Return the top-k most relevant documents
    top_documents = [item[0] for item in sorted_results[:k]]
    return top_documents
