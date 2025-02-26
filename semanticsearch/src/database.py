import os


class Database:
    def __init__(self, folder_path):
        """
        Initializes the database by loading all text files from the given folder.
        It stores the relative paths of the text files.

        Args:
            folder_path (str): Path to the folder containing text files.
        """
        self.folder_path = folder_path
        self.documents = []  # List of relative paths to the .txt files
        self._load_documents()

    def _load_documents(self):
        """
        Reads all .txt files in the folder and stores their relative paths.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder '{self.folder_path}' does not exist.")

        # Traverse through the folder and add all .txt file paths to the documents list
        for root, _, files in os.walk(self.folder_path):
            for filename in files:
                if filename.endswith(".txt"):
                    relative_path = os.path.relpath(os.path.join(root, filename), self.folder_path)
                    self.documents.append(relative_path)

    def get_document(self, relative_path):
        """
        Retrieves the content of a document given its relative path.

        Args:
            relative_path (str): The relative path of the document (from 'data/' folder).

        Returns:
            str: The content of the document, or None if not found.
        """
        file_path = os.path.join(self.folder_path, relative_path)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return None

    def list_documents(self):
        """
        Returns a list of all available document relative paths.

        Returns:
            list: A list of relative file paths to the .txt documents.
        """
        return self.documents
