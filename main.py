import tkinter as tk
from semanticsearch.src.semantic_search_app import SemanticSearchApp
from semanticsearch.src.reranking import Reranker


def main():
    re_ranking_system = Reranker()
    root = tk.Tk()
    SemanticSearchApp(root, width=40, re_ranking_system=re_ranking_system)
    root.mainloop()


if __name__ == "__main__":
    main()