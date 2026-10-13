"""Load Embedding Model names and Reranking Model names"""


def load_names(path):
    with open(path, 'r') as f:
        names = f.readlines()
    for i in range(len(names)):
        names[i] = names[i].split('#')[0].strip()
    names = [name for name in names if name]
    return names


def load_emb_model_names(path='embedding_models.txt'):
    return load_names(path)


def load_reranking_model_names(path='reranking_models.txt'):
    return load_names(path)
