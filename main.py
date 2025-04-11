import tkinter as tk
from tkinter import scrolledtext
from semanticsearch.src.inference import PageRecommender
from semanticsearch.src.misc import pprint
from semanticsearch.src.reranking import Reranker
from pprint import pformat


class SemanticSearchApp:
    def __init__(self, master, data_path='data\\raw', emb_file='data\\embeddings.json',
                 n_pages=4, width=50, max_text_length=2000, re_ranking_system=None):
        """

        :param master:
        :param data_path:
        :param emb_file:
        :param n_pages:
        :param width:
        :param max_text_length:
        :param re_ranking_system: takes as input a query and a list of documents,
            and re-orders them from most relevant to least relevant
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

        self.query_entry = tk.Entry(master, width=60)
        self.query_entry.pack(pady=5)

        self.search_button = tk.Button(master, text="Search", command=self.run_search)
        self.search_button.pack(pady=5)

        self.output_box = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=60, height=20)
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
            assert self.re_ranking_system is not None, 'Please provide re_ranking_system'
            recom_docs = [self.get_document(path) for path in recom_paths]
            best_doc = self.re_ranking_system.doc_rerank(query, recom_docs)[0]
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


def main():
    re_ranking_system = Reranker()
    root = tk.Tk()
    SemanticSearchApp(root, re_ranking_system=re_ranking_system)
    root.mainloop()


if __name__ == "__main__":
    main()