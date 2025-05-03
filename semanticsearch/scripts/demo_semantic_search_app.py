import tkinter as tk
from semanticsearch.src.semantic_search_app import SemanticSearchApp


def main():

    master = tk.Tk()
    SemanticSearchApp(master,
                      n_pages=5,
                      width=60,
                      re_ranking_system=None)
    master.mainloop()


if __name__ == "__main__":
    main()
