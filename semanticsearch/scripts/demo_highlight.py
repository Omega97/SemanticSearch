from langchain.text_splitter import RecursiveCharacterTextSplitter
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.database import Database
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b)


def add_green_background(text, green_intensity, base=38):
    """
    Prints text with a green background of arbitrary intensity.

    Args:
        text (str): The text to print.
        green_intensity (int): Green channel intensity (0-1).
    """
    # ANSI escape sequence for setting background color:
    # \033[48;2;<r>;<g>;<b>m - sets background to RGB(r,g,b)
    # Here, r=0, g=green_intensity, b=0
    green_background = f"\033[48;2;{base};{int(base + green_intensity*(256-base))};{base}m"
    reset = "\033[0m"

    return f"{green_background}{text}{reset}"


def main():
    # load query
    query = 'monte-carlo tree-search'

    # load document text
    db = Database('..\\..\\data\\raw')
    names = db.list_documents()
    name = names[0]
    text = db.get_document(name)

    # chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    docs = text_splitter.create_documents([text])
    chunks_page_content = [doc.page_content for doc in docs]

    # embedding model
    model = EmbeddingModel()

    # query embedding
    query_embedding = model.encode([query])

    # chunks embedding
    chunks_embeddings = model.encode(chunks_page_content)

    # cosine similarity
    similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunks_embeddings]

    # print highlighted chunks
    print(f'\nQUERY: {query}\n')
    for chunk, score in zip(chunks_page_content, similarities):
        score = score[0]
        score = max(0, score)**2
        print(add_green_background(chunk, score))


if __name__ == '__main__':
    main()
