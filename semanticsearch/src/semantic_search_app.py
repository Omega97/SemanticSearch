import tkinter as tk
from tkinter import scrolledtext
from pprint import pformat
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
                 n_pages=4, width=50, max_text_length=2000, re_ranking_system=None):
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

        # Create and place widgets
        self.query_label = tk.Label(master, text="Enter your query:")
        self.query_label.pack(pady=(10, 0))

        self.query_entry = tk.Entry(master, width=width)
        self.query_entry.pack(pady=5)

        self.search_button = tk.Button(master, text="Search", command=self.run_search)
        self.search_button.pack(pady=5)

        self.output_box = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=width, height=20)
        self.output_box.pack(pady=10)

    def get_document(self, file_name: str) -> str:
        return self.recommender.get_document(file_name)

    def run_search(self):
        query = self.query_entry.get().strip()
        if not query:
            return

        # Get recommendations
        recom_paths = self.recommender.recommend(query)
        print(recom_paths)

        # Fetch best document
        if self.re_ranking_system is not None:
            # Re-ranking
            docs_text = [self.get_document(path) for path in recom_paths]
            reordered_docs_text = self.re_ranking_system(query, docs_text)
            best_doc = reordered_docs_text[0]
            assert type(best_doc) is str
        else:
            # No re-ranking
            file_name = recom_paths[0]
            best_doc = self.get_document(file_name)

        # Size limit
        best_doc = best_doc[:self.max_text_length]

        # Reset the output widget and display the top document text in chunks
        self.output_box.delete('1.0', tk.END)
        self.output_box.insert(tk.END, f"\n{recom_paths[0]}\n\n")
        self.output_box.insert(tk.END, f"{pformat(best_doc, self.width)}")
        other_paths = '\n'.join(recom_paths[1:])
        self.output_box.insert(tk.END, f"\nSee also:\n{other_paths}")
