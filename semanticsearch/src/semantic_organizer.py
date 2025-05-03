import os
import shutil
from sklearn.cluster import KMeans
from semanticsearch.src.database import Database
from semanticsearch.src.embedding import EmbeddingModel, Embeddings


class SemanticFileOrganizer:
    """
    Organize the files in a directory by applying k-means
    on the semantic embedding of the files.
    """
    def __init__(self, root_directory, num_clusters, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the SemanticFileOrganizer.

        Args:
            root_directory (str): The root directory where files are located and will be rearranged.
            model_name (str): The name of the SentenceTransformer model to use for embeddings.
            num_clusters (int): The number of clusters to create.
        """
        self.root_directory = root_directory
        self.num_clusters = num_clusters
        self.model_name = model_name
        self.embeddings = None
        self.names = None
        self.kmeans = None
        self.embedding_matrix = None
        self.centroids = None
        self.folder_names = None

    def _flatten_directory(self):
        """
        Moves all files from subdirectories into the root directory and deletes empty directories.
        """
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(self.root_directory, file)
                # Avoid overwriting files with the same name
                if os.path.exists(destination_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_name = f"{base}_{counter}{ext}"
                        destination_path = os.path.join(self.root_directory, new_name)
                        counter += 1
                shutil.move(source_path, destination_path)

        # Delete empty directories
        for root, dirs, _ in os.walk(self.root_directory, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):  # Check if directory is empty
                    os.rmdir(dir_path)

    @staticmethod
    def _find_closest_embedding(centroid, embedding_matrix):
        """
        Finds the index of the embedding closest to the given centroid.

        Args:
            centroid (np.ndarray): The centroid vector.
            embedding_matrix (np.ndarray): The matrix of embeddings.

        Returns:
            int: The index of the closest embedding.
        """
        distances = ((embedding_matrix - centroid) ** 2).sum(axis=1)
        return distances.argmin()

    def _set_up_embeddings(self):
        database = Database(folder_path=self.root_directory)
        embedding_model = EmbeddingModel(model_name=self.model_name)
        self.embeddings = Embeddings(dir_path=self.root_directory, database=database, embedding_model=embedding_model)

    def _clustering(self):
        self.names, self.embedding_matrix = self.embeddings.to_matrix()
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(self.embedding_matrix)
        self.centroids = self.kmeans.cluster_centers_

    def _compute_folder_names(self):
        self.folder_names = []
        for centroid in self.centroids:
            closest_index = self._find_closest_embedding(centroid, self.embedding_matrix)
            closest_file = self.names[closest_index]
            folder_name = os.path.splitext(closest_file)[0]  # Remove file extension
            self.folder_names.append(folder_name)

    def _create_folders(self):
        for folder_name in self.folder_names:
            os.makedirs(os.path.join(self.root_directory, folder_name), exist_ok=True)

    def _move_files(self):
        labels = self.kmeans.labels_
        for file_name, label in zip(self.names, labels):
            source_path = os.path.join(self.root_directory, file_name)
            destination_folder = self.folder_names[label]
            destination_path = os.path.join(self.root_directory, destination_folder, file_name)
            shutil.move(source_path, destination_path)

    def organize_files(self):
        """
        Rearranges files in the root directory into subfolders based on semantic embeddings.
        """
        # Step 1: Move all files to the root directory and delete other directories
        print('Flattening directory...')
        self._flatten_directory()

        # Step 2: Set up the Embeddings
        print('Setting up embeddings...')
        self._set_up_embeddings()

        # Step 3: Perform clustering and get centroids
        print('Clustering...')
        self._clustering()

        # Step 4: For each centroid, compute the closest file and use it as the folder name
        print('Computing folder names...')
        self._compute_folder_names()

        # Step 5: Create new directories
        print('Creating new folders...')
        self._create_folders()

        # Step 6: Move files to the new directories based on clustering
        print('Moving files...')
        self._move_files()

        # Delete embeddings
        self.embeddings.delete_embeddings_file()

        print("Files have been successfully organized into semantic folders.")
