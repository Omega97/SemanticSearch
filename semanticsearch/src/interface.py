import streamlit as st
import json
import numpy as np
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.database import Database


def find_relevant_documents(query, embedding_model, embeddings_data):
    """ Function to compute the similarity between the query and documents """
    # Compute the query embedding
    query_embedding = embedding_model.encode([query])[0]

    # Compute cosine similarity between the query embedding and each document's embedding
    similarities = {}
    for file_path, embedding in embeddings_data.items():
        embedding = np.array(embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities[file_path] = similarity

    # Sort by similarity in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    # Return the top 5 relevant documents
    top_documents = [item[0] for item in sorted_similarities[:5]]
    return top_documents


def main():
    # Load the database and embedding model
    database = Database('data/')
    embedding_model = EmbeddingModel()

    # Load existing embeddings (if available)
    embeddings_file = 'data/embeddings.json'
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    # Streamlit UI
    st.title("Semantic Search Interface")

    # Input query from the user
    query = st.text_input("Enter your query:")

    if query:
        # Find the most relevant documents
        relevant_documents = find_relevant_documents(query, embedding_model, embeddings_data)

        # Display the results
        st.write("### Relevant Documents:")
        for doc in relevant_documents:
            st.write(f"- {doc}")


if __name__ == '__main__':
    main()
