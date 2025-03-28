import tkinter as tk
from tkinter import scrolledtext
from semanticsearch.src.inference import PageRecommender
from semanticsearch.src.misc import pprint


class SemanticSearchApp:
    def __init__(self, master, data_path='data\\raw', emb_file='data\\embeddings.json',
                 n_pages=4, width=50, max_text_length=2000):
        self.master = master
        self.master.title("Semantic Search")

        # Instantiate the recommendation system
        self.pr = PageRecommender(data_path, emb_file, k=n_pages)
        self.width = width
        self.max_text_length = max_text_length

        # Create and place widgets
        self.query_label = tk.Label(master, text="Enter your query:")
        self.query_label.pack(pady=(10, 0))

        self.query_entry = tk.Entry(master, width=60)
        self.query_entry.pack(pady=5)

        self.search_button = tk.Button(master, text="Search", command=self.run_search)
        self.search_button.pack(pady=5)

        self.output_box = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=60, height=20)
        self.output_box.pack(pady=10)

    def run_search(self):
        query = self.query_entry.get().strip()
        if not query:
            return

        # Get recommendations
        recommendations = self.pr.recommend(query)

        # Reset the output widget
        self.output_box.delete('1.0', tk.END)

        # Fetch and display the top document text in chunks
        file_name = recommendations[0]
        doc = self.pr.get_document(file_name)
        self.output_box.insert(tk.END, f"\n{recommendations[0]}\n\n")
        self.output_box.insert(tk.END, f"{pprint(doc[:self.max_text_length], self.width)}\n")
        other = '\n'.join(recommendations[1:])
        self.output_box.insert(tk.END, f"\nSee also:\n{other}")


def main():
    root = tk.Tk()
    SemanticSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
