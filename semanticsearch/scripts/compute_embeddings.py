import os
import json


def compute_embeddings(database, embedding_model, embeddings_file):

    # List all the documents that require embeddings
    existing_embeddings = {}

    # Check if embeddings already exist to avoid recomputing them
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as f:
            existing_embeddings = json.load(f)
        print("Loaded existing embeddings from 'data/embeddings.json'.")
        print(f"Documents pre-embedded: {len(existing_embeddings)}")

    # Filter out documents that already have embeddings
    documents_to_embed = [doc for doc in database.documents if doc not in existing_embeddings]

    # Compute embeddings for documents that need them
    embeddings_batch = embedding_model.encode(documents_to_embed)

    # Prepare the data for saving (file path and corresponding embeddings)
    new_embeddings = [
        {"file_path": doc_path, "embedding": [f"{x:.6e}" for x in embedding]}  # Convert to scientific notation
        for doc_path, embedding in zip(documents_to_embed, embeddings_batch)
    ]

    # Step 5: Merge new embeddings with existing ones
    existing_embeddings.update({doc['file_path']: doc['embedding'] for doc in new_embeddings})

    # Step 6: Save embeddings to JSON file
    with open(embeddings_file, 'w') as f:
        json.dump(existing_embeddings, f, indent=4)

    print(f"Embeddings saved to {embeddings_file}")
    print(f"Total documents: {len(database.documents)}")
