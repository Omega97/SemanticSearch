""" semanticsearch/src/database.py

The Database class handles the documents to retrieve
in the root directory.
"""
import os
from semanticsearch.src.misc import get_extension, read_txt_file, read_csv_file


# ----- Default Parameters -----
DEFAULT_FILE_READERS = {'txt': read_txt_file,
                        'csv': read_csv_file}


class Database:
    """
    This class can find all the documents of a given type in the root folder and in
    all the subdirectories. Then, you can read a file by selecting its path.
    """
    def __init__(self, folder_path, file_readers=None):
        """
        Initializes the database by loading all text files from the given folder.
        It stores the relative paths of the text files.

        Args:
            folder_path (str): Path to the folder containing text files.
            file_readers (dict, optional): A dictionary mapping file extensions to reader functions.
                                           Defaults to DEFAULT_FILE_READERS.
        """
        self.folder_path = folder_path

        # Initialize file_readers safely
        if file_readers is None:
            self.file_readers = DEFAULT_FILE_READERS.copy()  # Create a shallow copy to avoid mutation
        else:
            self.file_readers = file_readers

        self.documents = []  # List of relative paths to the .txt files
        self._load_documents()

    def get_relative_path(self, root, filename):
        """Return path relative to root"""
        return os.path.relpath(os.path.join(root, filename), self.folder_path)

    def _load_documents(self):
        """
        Scans through all .txt files in the folder and stores their relative paths.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder '{self.folder_path}' does not exist.")

        # Traverse through the folder and add all .txt file paths to the documents list
        for root, _, files in os.walk(self.folder_path):
            for filename in files:
                if get_extension(filename) in self.file_readers:
                    rel_path = self.get_relative_path(root, filename)
                    self.documents.append(rel_path)

    def get_document(self, relative_path, **kwargs):
        """
        Retrieves the content of a document given its relative path.

        Args:
            relative_path (str): The relative path of the document (from 'data/' folder).

        Returns:
            str: The content of the document
        """
        file_path = os.path.join(self.folder_path, relative_path)
        ext = get_extension(file_path)
        return self.file_readers[ext](file_path, **kwargs)

    def list_documents(self):
        """
        Returns a list of all available document relative paths.

        Returns:
            list: A list of relative file paths to the .txt documents.
        """
        return tuple(self.documents)

    def iter_documents(self):
        for path in self.list_documents():
            yield {'path': path, 'text': self.get_document(path)}
