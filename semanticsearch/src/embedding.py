""" semanticsearch/src/embedding.py

This module contains the EmbeddingModel class, which is used to convert
text inputs into vector embeddings. The class uses the SentenceTransformer
library to encode the text inputs.

Useful models:
- all-MiniLM-L6-v2 (small & efficient)
- paraphrase-MiniLM-L6-v2 (small & efficient)
- stsb-distilroberta-base-v2 (small)
- paraphrase-mpnet-base-v2 (medium)
- paraphrase-TinyBERT-L6-v2 (medium)
- paraphrase-distilroberta-base-v1 (medium)
- paraphrase-multilingual-mpnet-base-v2 (multilingual)
- paraphrase-xlm-r-multilingual-v1 (multilingual)
"""
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from semanticsearch.src.database import Database
from semanticsearch.src.misc import cprint


# ----- Default Parameters -----
DEFAULT_EMBEDDINGS_NAME = "_embeddings.npz"


class EmbeddingModel:
    """
    The EmbeddingModel class is used to convert text inputs into vector embeddings.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        cprint(f'Loading embedding model f{model_name}', 'b')
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list, normalize_embeddings=True) -> np.array:
        """
        Converts a list of text inputs into vector embeddings.

        Args:
            texts (list of str): List of text inputs.
            normalize_embeddings (bool): Whether to normalize the embeddings.

        Returns:
            np.ndarray: Array, the embeddings matrix (shape: [num_texts, embedding_dim]).
        """
        return np.array(self.model.encode(texts,
                                          normalize_embeddings=normalize_embeddings,
                                          convert_to_numpy=True))

    def __repr__(self):
        return f'{type(self)}({self.model_name})'


class Embeddings:
    """
    Generate file embeddings, save them, and load them when needed.
    You can also extract the file names as a list and the embeddings
    as a matrix with *to_matrix()*.
    """
    def __init__(self, dir_path: str,
                 database: Database,
                 embedding_model: EmbeddingModel,
                 data_file_name=DEFAULT_EMBEDDINGS_NAME):
        """
        Initializes the Embeddings class with a directory path.

        Args:
            dir_path (str): Path to the directory where embeddings are stored/loaded.
            embedding_model: embedding_model.encode(texts) returns the matrix of embeddings
        """
        self.dir_path = dir_path
        self.database = database
        self.embedding_model = embedding_model
        self.data_file_name = data_file_name
        self.data = {}  # Dictionary to store {name: embedding}
        self.missing_embeddings = None
        self._load()

    def get_data_file_name(self) -> str:
        return os.path.join(self.dir_path, self.data_file_name)

    def _setup(self):
        """ Set up the environment. """
        # Ensure the directory exists
        os.makedirs(self.dir_path, exist_ok=True)

        # Ensure the data file exists
        if not os.path.exists(self.get_data_file_name()):
            self._save()

    def _load_file_paths_from_database(self):
        self.all_paths = self.database.list_documents()

    def _load_embeddings_from_file(self):
        print(f'Loading embeddings from {self.get_data_file_name()}')
        try:
            data = np.load(self.get_data_file_name(), allow_pickle=True)
            # change "/" to "\\"
            data = {key.replace("/", "\\"): value for key, value in data.items()}

            # intersect data and all_paths
            self.data = {key: data[key] for key in data if key in self.all_paths}

            cprint(f"Loaded {len(self.data)} embeddings from {self.get_data_file_name()}", "g")
        except Exception as e:
            cprint(f"{e}\nFailed to load embeddings, starting fresh.", "r")
            self.data = {}

    def _find_missing_embeddings(self):
        """
        Identifies documents in the database that do not yet have embeddings.
        """
        all_paths_set = set(self.all_paths)
        existing_keys_set = set(self.data.keys())
        self.missing_embeddings = list(all_paths_set - existing_keys_set)

    def _compute_missing_embeddings(self):
        if not self.missing_embeddings:
            print("No new documents to embed.")
            return

        texts = [self.database.get_document(p) for p in self.missing_embeddings]
        embeddings = self.embedding_model.encode(texts)
        print(f'Extending with {len(embeddings)} embeddings')
        self.extend(self.missing_embeddings, embeddings)
        self._save()

    def _load(self):
        """
        Loads the embeddings from disk into the instance attribute `self.data`.
        """
        # Set up the environment
        self._setup()

        # Load file paths from Database
        self._load_file_paths_from_database()

        # Load existing embeddings
        self._load_embeddings_from_file()

        # Find out what embeddings are missing
        self._find_missing_embeddings()

        # Compute missing embeddings
        self._compute_missing_embeddings()

    def _save(self):
        """
        Saves the embeddings from the instance attribute `self.data` to disk.
        """
        np.savez(file=self.get_data_file_name(), **self.data)
        if len(self.data):
            cprint(f"Saved {len(self.data)} embeddings to {self.get_data_file_name()}", "green")

    def extend(self, names: list[str], embeddings: np.ndarray):
        """
        Extends the embeddings dictionary with new data.

        Args:
            names (list[str]): List of file names to add.
            embeddings (np.ndarray): New embeddings matrix to append.

        Raises:
            ValueError: If the number of names does not match the number of embeddings.
        """
        if len(names) != embeddings.shape[0]:
            raise ValueError("The number of names must match the number of embedding rows.")

        # Add new entries to the dictionary
        for name, embedding in zip(names, embeddings):
            if name in self.data:
                print(f"Warning: Overwriting existing entry for '{name}'.")
            self.data[name] = embedding

        cprint(f"Extended embeddings to {len(self.data)} entries.", "g")

    def remove(self, names: list[str]):
        """
        Removes specified file names and their corresponding embeddings.

        Args:
            names (list[str]): List of file names to remove.

        Raises:
            ValueError: If any of the names are not found in the current dictionary.
        """
        if not isinstance(names, list):
            raise TypeError("The 'names' argument must be a list of strings.")

        missing_names = [name for name in names if name not in self.data]
        if missing_names:
            raise ValueError(f"The following names were not found: {missing_names}")

        # Remove entries
        for name in names:
            del self.data[name]

        cprint(f"Removed {len(names)} entries.", "r")

    def get(self, name: str) -> np.ndarray:
        """
        Retrieves the embedding vector for a specific file name.

        Args:
            name (str): The file name to retrieve the embedding for.

        Returns:
            np.ndarray: The embedding vector.

        Raises:
            KeyError: If the name is not found in the dictionary.
        """
        if name not in self.data:
            raise KeyError(f"No embedding found for '{name}'.")
        return self.data[name]

    def to_matrix(self) -> tuple[list[str], np.ndarray]:
        """
        Converts the dictionary into a list of names and a matrix of embeddings.

        Returns:
            tuple[list[str], np.ndarray]: A tuple containing the list of names and the embeddings matrix.
        """
        names = list(self.data.keys())
        embeddings = np.array(list(self.data.values()))
        return names, embeddings
