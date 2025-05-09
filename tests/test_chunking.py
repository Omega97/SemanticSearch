import tkinter as tk
from semanticsearch.src.semantic_search_app import SemanticSearchApp
from semanticsearch.src.reranking import Reranker


def main():
    re_ranking_system = Reranker(chunking_enabled=True,
                                 chunk_size=1000,
                                 max_n_chunks=5)

    root = tk.Tk()
    SemanticSearchApp(root,
                      data_path='..\\data\\raw',
                      emb_file='..\\data\\embeddings.json',
                      n_pages=5,
                      width=60,
                      re_ranking_system=re_ranking_system)
    root.mainloop()


if __name__ == "__main__":
    main()
