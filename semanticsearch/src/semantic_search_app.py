import tkinter as tk
from tkinter import scrolledtext
from semanticsearch.src.inference import PageRecommender


# --- Constants ---
DEFAULT_DATA_PATH = 'data\\raw'
DEFAULT_EMB_FILE = 'data\\embeddings.json'


class SemanticSearchApp:
    """
    A Tkinter GUI application for performing semantic search on documents
    and displaying the most relevant results, with optional re-ranking.
    """
    def __init__(self, master, data_path=DEFAULT_DATA_PATH, emb_file=DEFAULT_EMB_FILE,
                 n_pages=5, width=50, max_text_length=2000, re_ranking_system=None):
        """
        Initializes the Semantic Search application interface.

        :param master: tk.Tk root
        :param data_path: directory with all the documents to perform retrieval on
        :param emb_file: file with the embeddings
        :param n_pages: number of pages to retrieve for the re-ranking system
        :param width: line width (for display purposes)
        :param max_text_length: maximum length of a document number of characters
        :param re_ranking_system: takes as input a query and a list of documents,
            and re-orders them from the most relevant to the least relevant
        """
        self.master = master
        self.master.title("Semantic Search")

        # Instantiate the recommendation system
        self.recommender = PageRecommender(data_path, emb_file, k=n_pages)
        self.width = width
        self.max_text_length = max_text_length
        self.re_ranking_system = re_ranking_system

        self.recom_paths = None

        # Initialize widgets
        self.query_label = None
        self.query_entry = None
        self.search_button = None
        self.output_box = None

        self.place_widgets()

    def place_widgets(self, pady=5):
        """Create and place widgets"""
        self.query_label = tk.Label(self.master, text="Enter your query:")
        self.query_label.pack(pady=(2 * pady, 0))

        self.query_entry = tk.Entry(self.master, width=self.width)
        self.query_entry.pack(pady=pady)

        self.search_button = tk.Button(self.master, text="Search", command=self.run_search)
        self.search_button.pack(pady=pady)

        self.output_box = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=self.width, height=20)
        self.output_box.pack(pady=2 * pady)

    def get_document(self, file_name: str) -> str:
        """Returns the text of the document of the given file path"""
        return self.recommender.get_document(file_name)

    def get_best_document(self) -> str:
        """Return the best document"""
        file_name = self.recom_paths[0]
        doc = self.get_document(file_name)
        doc = doc[:self.max_text_length]
        return doc

    def print_recommended_paths(self, title=None):
        """Print paths of recommended documents, from most to least recommended."""
        if title:
            print(title)
        for path in self.recom_paths:
            print(f'* {path}')

    def _do_re_ranking(self, query):
        """
        Assign best documents to self.best_doc.
        If re_ranking_system is loaded, do re-ranking first.
        """
        if self.re_ranking_system is not None:
            # Re-ranking
            docs_text = [self.get_document(path) for path in self.recom_paths]
            permutation = self.re_ranking_system(query, docs_text)

            # Change the order of paths
            self.recom_paths = [self.recom_paths[i] for i in permutation]
            self.print_recommended_paths('Top docs after re-ranking')

    def get_query(self) -> str:
        query = self.query_entry.get().strip()
        print(f'\n>>> {query}')
        return query

    def display_doc_on_window(self):
        self.output_box.delete('1.0', tk.END)
        self.output_box.insert(tk.END, f"\n{self.recom_paths[0]}\n\n")
        self.output_box.insert(tk.END, self.get_best_document())
        other_paths = '\n'.join(self.recom_paths[1:])
        self.output_box.insert(tk.END, f"\n\nSee also:\n{other_paths}")

    def compute_recommended_paths(self, query):
        """Get list of paths of recommended documents"""
        self.recom_paths = self.recommender.recommend(query)
        self.print_recommended_paths('Top docs:')

    def run_search(self):
        """Interface for semantic retrieval"""
        # Get query
        query = self.get_query()

        # Get recommendations
        self.compute_recommended_paths(query)

        # Fetch best document
        self._do_re_ranking(query)

        # Reset the output widget and display the top document text in chunks
        self.display_doc_on_window()
